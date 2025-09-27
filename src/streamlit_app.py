#!/usr/bin/env python3
"""
Streamlit UI for Social Support AI — fixed explainer parsing, XGBoost display, no duplicate outputs.

Behavior:
- Background orchestrator thread writes logs to a queue; main thread drains on Refresh.
- Explainer subprocess stdout is parsed for a JSON object followed by human text.
  - JSON fields used: final_decision, category, xgb_prob (or xgb_result.prob), short_bullets, recommended_action
- Displays metrics: Final decision (APPROVE/REJECT) and XGBoost probability.
- Avoids appending duplicate assistant messages to conversation history.
- Quick buttons to prefill last-10 test applicant IDs (APP100041..APP100050).
"""

from __future__ import annotations
import os
import sys
import json
import threading
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import queue
import re

import streamlit as st

# Config
DEFAULT_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_MONGO_DB = os.getenv("MONGO_DB", "socialsupport")
PYTHON_EXEC = os.getenv("PYTHON_EXEC", sys.executable)
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"

# Thread-safe queue for logs
_log_queue: "queue.Queue[str]" = queue.Queue()

# Session state initialization (main thread)
if "orch_running" not in st.session_state:
    st.session_state["orch_running"] = False
if "orch_logs" not in st.session_state:
    st.session_state["orch_logs"] = []
if "orch_run_id" not in st.session_state:
    st.session_state["orch_run_id"] = None
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {}
if "last_profile" not in st.session_state:
    st.session_state["last_profile"] = {}
if "input_applicant" not in st.session_state:
    st.session_state["input_applicant"] = ""

# Utilities
def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _queue_put(line: str):
    try:
        _log_queue.put_nowait(line)
    except Exception:
        pass

def drain_log_queue(max_lines: int = 2000) -> int:
    moved = 0
    while True:
        try:
            line = _log_queue.get_nowait()
            st.session_state["orch_logs"].append(line)
            moved += 1
            if moved >= max_lines:
                break
        except queue.Empty:
            break
    return moved

# Subprocess helpers
def stream_process_capture(cmd: List[str], cwd: Optional[str] = None, append_fn=None, timeout: Optional[int] = None) -> int:
    if append_fn is None:
        append_fn = lambda s: None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            env=os.environ,
            text=True,
        )
    except Exception as e:
        append_fn(f"[ERROR] failed to start process: {e}\n")
        return 2

    def _reader(stream, prefix=""):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                append_fn(prefix + line)
        except Exception as e:
            append_fn(f"[ERROR] reader error: {e}\n")

    t_out = threading.Thread(target=_reader, args=(proc.stdout, ""), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "[ERR] "), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        append_fn("[ERROR] process timeout and was killed\n")
        return 2

    time.sleep(0.05)
    return proc.returncode

# Background orchestrator thread
def orchestrator_thread_fn(limit: int = 50, llm_model: str = "mistral"):
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    _queue_put(f"--- Orchestrator run_id={run_id} started at {now_iso()} ---\n")
    cmd = [PYTHON_EXEC, str(SRC_DIR / "orchestrator.py"), "--limit", str(limit), "--llm-model", llm_model]
    _queue_put(f"[CMD] {' '.join(cmd)}\n")
    rc = stream_process_capture(cmd, cwd=str(PROJECT_ROOT), append_fn=_queue_put, timeout=None)
    _queue_put(f"--- Orchestrator finished rc={rc} run_id={run_id} at {now_iso()} ---\n")
    _queue_put(f"__ORCH_THREAD_COMPLETE__:{run_id}:{rc}\n")

def start_orchestrator(limit: int = 50, llm_model: str = "mistral"):
    if st.session_state["orch_running"]:
        st.warning("Orchestrator already running.")
        return
    st.session_state["orch_logs"] = []
    while True:
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break
    st.session_state["orch_running"] = True
    t = threading.Thread(target=orchestrator_thread_fn, args=(limit, llm_model), daemon=True)
    t.start()
    st.success("Orchestrator started; press Refresh logs to pull output.")

# Mongo helper
def fetch_canonical_profile(applicant_id: str, mongo_uri: str = DEFAULT_MONGO_URI, mongo_db: str = DEFAULT_MONGO_DB) -> Optional[Dict[str, Any]]:
    try:
        from pymongo import MongoClient
    except Exception:
        return None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[mongo_db]
        doc = db.canonical_profiles.find_one({"applicant_id": applicant_id}, {"_id": 0})
        return doc
    except Exception:
        return None

# Explainer subprocess call (returns raw stdout/stderr+rc)
def call_llm_explainer_subprocess(applicant_id: str, model: str = "mistral", dry_run: bool = True, timeout: int = 120) -> Dict[str, Any]:
    cmd = [PYTHON_EXEC, str(SRC_DIR / "llm_explainer_mistral.py"), "--applicant", applicant_id]
    if not dry_run:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(PROJECT_ROOT), env=os.environ.copy(), timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as e:
        return {"returncode": 2, "stdout": "", "stderr": f"TimeoutExpired: {e}"}
    except Exception as e:
        return {"returncode": 2, "stdout": "", "stderr": f"Exception: {e}"}

# Parsing helper: extract JSON object (first balanced {...}) and remaining human text
def parse_json_and_human(stdout: str) -> Tuple[Optional[dict], Optional[str]]:
    if not stdout:
        return None, None
    s = stdout.strip()
    # find first balanced JSON object: naive but effective for one top-level object
    first = s.find("{")
    if first == -1:
        return None, s
    # attempt to find matching closing brace by scanning char-by-char counting braces
    depth = 0
    start = first
    end = None
    for i in range(first, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        # not found balanced braces
        return None, s
    json_snip = s[start:end+1]
    rest = s[end+1:].strip()
    try:
        parsed = json.loads(json_snip)
        human = rest if rest else None
        return parsed, human
    except Exception:
        # fallback: try to safe-evaluate by searching for first/last brace and loading via json5 if available
        try:
            import json5  # type: ignore
            parsed = json5.loads(json_snip)
            human = rest if rest else None
            return parsed, human
        except Exception:
            return None, s

# Avoid duplicate assistant entries
def append_assistant_once(applicant_id: str, content: str):
    conv = st.session_state["conversations"].setdefault(applicant_id, [])
    # if last message equals content, skip
    if conv and conv[-1].get("role") == "assistant" and conv[-1].get("content") == content:
        return
    conv.append({"role": "assistant", "content": content})

# Streamlit UI layout
st.set_page_config(page_title="Social Support AI — Demo", layout="wide")
st.title("Social Support AI — Demo (Local) — Fixed explainer & XGB display")

st.markdown(
    "- Run orchestrator (background thread). Logs are queued and shown when you press **Refresh logs**.\n"
    "- Run explainer: parsed JSON will show `final_decision` and `xgb_prob` where available.\n"
    "- Quick applicant buttons fill the input for APP100041..APP100050 (testing last-10)."
)

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Orchestrator")
    limit = st.number_input("How many canonical profiles to create (limit)", min_value=1, max_value=1000, value=50, step=1)
    llm_model = st.text_input("LLM model (ollama)", value="mistral")
    run_btn = st.button("Run Orchestrator (start)", key="run_orch")
    refresh_btn = st.button("Refresh logs", key="refresh_logs")
    clear_logs_btn = st.button("Clear logs", key="clear_logs")

    if run_btn:
        try:
            start_orchestrator(limit=int(limit), llm_model=llm_model)
        except Exception as e:
            st.error(f"Failed to start orchestrator thread: {e}")
            st.exception(e)

    if refresh_btn:
        moved = drain_log_queue()
        tail = "".join(st.session_state["orch_logs"][-100:])
        if "__ORCH_THREAD_COMPLETE__" in tail or "__ORCH_THREAD_COMPLETE__" in tail:
            st.session_state["orch_running"] = False
            st.success(f"Drained {moved} lines. Orchestrator finished.")
        else:
            st.success(f"Drained {moved} lines into logs (latest at bottom).")

    if clear_logs_btn:
        st.session_state["orch_logs"] = []
        while True:
            try:
                _log_queue.get_nowait()
            except queue.Empty:
                break
        st.success("Logs cleared (session + background queue).")

    if st.session_state["orch_running"]:
        st.warning("Orchestrator appears to be running (background). Use Refresh logs to pull output.")
    else:
        st.info("Orchestrator is idle.")

    st.markdown("---")
    st.markdown("Quick test applicant IDs (last 10):")
    quick_ids = [f"APP1000{str(i).zfill(2)}" for i in range(41, 51)]
    qcols = st.columns(len(quick_ids))
    for cid, ccol in zip(quick_ids, qcols):
        if ccol.button(cid, key=f"quick_{cid}"):
            st.session_state["input_applicant"] = cid
            st.success(f"Set applicant id to {cid}")

with col2:
    st.header("Query applicant / LLM explainer")
    applicant_id = st.text_input("Applicant ID (e.g. APP100001)", value=st.session_state.get("input_applicant", ""), key="input_applicant")

    # actions
    ac1, ac2, ac3 = st.columns([1,1,1])
    with ac1:
        if st.button("Fetch canonical profile"):
            if not applicant_id.strip():
                st.warning("Enter an applicant_id first.")
            else:
                doc = fetch_canonical_profile(applicant_id.strip())
                if not doc:
                    st.error("No canonical profile found or cannot connect to Mongo.")
                else:
                    st.session_state["last_profile"][applicant_id.strip()] = doc
                    st.success(f"Fetched canonical profile for {applicant_id.strip()}")
    with ac2:
        if st.button("Clear conversation"):
            st.session_state["conversations"].pop(applicant_id.strip(), None)
            st.success("Conversation cleared.")
    with ac3:
        if st.button("Clear cached profile"):
            st.session_state["last_profile"].pop(applicant_id.strip(), None)
            st.success("Cleared cached profile.")

    # show profile if available
    profile = None
    if applicant_id.strip() and st.session_state["last_profile"].get(applicant_id.strip()):
        profile = st.session_state["last_profile"][applicant_id.strip()]
    elif applicant_id.strip():
        profile = fetch_canonical_profile(applicant_id.strip())
        if profile:
            st.session_state["last_profile"][applicant_id.strip()] = profile

    if profile:
        st.subheader("Canonical profile (fetched)")
        st.json(profile)
        st.write(f"**Validation status:** {profile.get('validation_status')} — **confidence_score:** {profile.get('confidence_score')}")
    else:
        st.info("No canonical profile loaded. Use 'Fetch canonical profile' or ensure the orchestrator has run.")

    st.markdown("---")
    st.markdown("#### LLM explainer (on-demand)")
    explainer_mode = st.radio("Explainer mode", ("subprocess script (recommended)", "direct ollama (interactive)"), index=0)
    use_real_llm = st.checkbox("Use real LLM (mistral via Ollama). If unchecked, runs deterministic dry-run.", value=False)
    explain_btn = st.button("Run LLM explainer now")

    if explain_btn:
        if not applicant_id.strip():
            st.warning("Enter an applicant_id first.")
        else:
            if explainer_mode.startswith("subprocess"):
                dry_run = not use_real_llm
                out = call_llm_explainer_subprocess(applicant_id.strip(), model=llm_model, dry_run=dry_run)
                # show stderr if any
                if out["stderr"]:
                    st.text_area("Explainer stderr (if any)", value=out["stderr"], height=120)
                if out["returncode"] != 0:
                    st.error(f"Explainer script exited rc={out['returncode']}")
                if out["stdout"]:
                    # parse JSON + human
                    parsed_json, human = parse_json_and_human(out["stdout"])
                    # if parsed_json present, show metrics and structured info
                    if parsed_json:
                        # display final decision and xgb prob
                        final = parsed_json.get("final_decision") or parsed_json.get("category")
                        # xgb prob: look in multiple possible fields
                        xgb_prob = None
                        if "xgb_prob" in parsed_json:
                            xgb_prob = parsed_json["xgb_prob"]
                        elif isinstance(parsed_json.get("xgb_result"), dict):
                            xgb_prob = parsed_json["xgb_result"].get("prob")
                        elif isinstance(parsed_json.get("xgb_result"), (list,)):
                            xgb_prob = parsed_json.get("xgb_result")
                        # show top-level info
                        st.success("Explainer produced structured JSON and human summary.")
                        st.subheader("Parsed JSON")
                        st.json(parsed_json)
                        # Metrics
                        if final:
                            st.metric("Final decision", value=str(final).upper())
                        if xgb_prob is not None:
                            try:
                                st.metric("XGBoost prob", value=f"{float(xgb_prob):.3f}")
                            except Exception:
                                st.write("XGBoost prob:", xgb_prob)
                        # Show bullets if present
                        bullets = parsed_json.get("short_bullets") or parsed_json.get("short_bullets", [])
                        if bullets:
                            st.write("Short bullets:")
                            for b in bullets:
                                st.write("- " + str(b))
                        # recommended action
                        action = parsed_json.get("recommended_action") or parsed_json.get("recommended_action", "")
                        if action:
                            st.write("Recommended action:")
                            st.write(action)
                    else:
                        # no JSON parsed - show raw stdout
                        st.info("Explainer output (raw):")
                        st.text_area("Explainer stdout (raw)", value=out["stdout"], height=300)

                    # Human/paragraph display (prefer extracted human paragraph if present)
                    if parsed_json and human:
                        # Some explainer implementations return the human paragraph after the JSON — show it
                        st.subheader("Human summary (from explainer):")
                        st.write(human)
                        assistant_content = out["stdout"]  # store full stdout
                    elif not parsed_json:
                        assistant_content = out["stdout"]
                    else:
                        # parsed_json present but no human text — try to find a human-friendly field in JSON
                        human_text = parsed_json.get("human") or parsed_json.get("human_text") or parsed_json.get("explanation")
                        if human_text:
                            st.subheader("Human summary:")
                            st.write(human_text)
                            assistant_content = out["stdout"]
                        else:
                            assistant_content = out["stdout"]

                    # append assistant message once
                    append_assistant_once(applicant_id.strip(), assistant_content)

            else:
                # direct ollama path
                try:
                    import ollama
                    doc = fetch_canonical_profile(applicant_id.strip())
                    if not doc:
                        st.error("No canonical profile found for applicant (cannot call ollama).")
                    else:
                        system_msg = {"role":"system", "content":"You are an expert decision explainer. Use ONLY the canonical_profile context provided."}
                        context_msg = {"role":"system", "content":"CANONICAL_PROFILE:\n" + json.dumps(doc, indent=2)[:4000]}
                        user_req = {"role":"user", "content":"Please produce a concise decision summary, evidence bullets, and a recommended action based on the canonical profile. Also include a JSON block with fields: category, final_decision, xgb_prob, short_bullets, recommended_action."}
                        messages = [system_msg, context_msg, user_req]
                        resp = ollama.chat(model=llm_model, messages=messages)
                        if isinstance(resp, dict):
                            content = resp.get("message",{}).get("content") or resp.get("content") or str(resp)
                        else:
                            content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
                        st.success("LLM assistant reply:")
                        st.write(content)
                        append_assistant_once(applicant_id.strip(), content)
                except Exception as e:
                    st.error(f"Ollama direct call failed: {e}")
                    st.exception(e)

    st.markdown("---")

    # Conversation / follow-ups
    st.markdown("### Conversation / follow-ups")
    conv = st.session_state["conversations"].get(applicant_id.strip(), [])
    if conv:
        for m in conv:
            who = "User" if m["role"] == "user" else "Assistant"
            st.markdown(f"**{who}:**")
            st.write(m["content"])
    else:
        st.info("No conversation yet for this applicant (run explainer first).")

    followup = st.text_input("Ask a follow-up question (will use conversation + profile context)", key=f"followup_{applicant_id}")
    ask_btn = st.button("Ask follow-up")

    if ask_btn:
        if not applicant_id.strip():
            st.warning("Enter applicant id first.")
        elif not followup.strip():
            st.warning("Enter a follow-up question.")
        else:
            try:
                import ollama
                doc = fetch_canonical_profile(applicant_id.strip())
                if not doc:
                    st.error("No canonical profile available for follow-up (fetch it first).")
                else:
                    system_msg = {"role":"system", "content":"You are an expert decision explainer. Use ONLY the canonical_profile context provided."}
                    context_msg = {"role":"system", "content":"CANONICAL_PROFILE:\n" + json.dumps(doc, indent=2)[:4000]}
                    history = st.session_state["conversations"].get(applicant_id.strip(), [])
                    # convert conversation into messages for ollama: keep roles and content
                    messages = [system_msg, context_msg]
                    for h in history:
                        # map our stored messages to chat messages
                        role = h.get("role", "assistant")
                        # user/assistant expected by ollama
                        messages.append({"role": role, "content": h.get("content")})
                    messages.append({"role":"user", "content": followup})
                    resp = ollama.chat(model=llm_model, messages=messages)
                    if isinstance(resp, dict):
                        content = resp.get("message",{}).get("content") or resp.get("content") or str(resp)
                    else:
                        content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
                    st.success("Assistant reply:")
                    st.write(content)
                    # append user then assistant (avoid duplicate)
                    st.session_state["conversations"].setdefault(applicant_id.strip(), []).append({"role":"user","content": followup})
                    append_assistant_once(applicant_id.strip(), content)
            except Exception as e:
                st.error(f"Direct Ollama call failed: {e}")
                st.exception(e)

# Bottom logs
st.markdown("---")
st.header("Orchestrator logs (bottom)")

# Do a small automatic drain on page load (non-blocking)
drain_log_queue(max_lines=100)

if st.session_state["orch_logs"]:
    log_text = "".join(st.session_state["orch_logs"][-4000:])
    st.text_area("Orchestrator logs", value=log_text, height=350)
else:
    st.info("No logs yet. Run orchestrator to see logs here. Use 'Refresh logs' to update.")

st.markdown("---")
st.caption("Streamlit UI — make sure your venv is active and the src/ scripts exist.")