#!/usr/bin/env python3
"""
orchestrator.py (updated to call upload_raw_to_mongo.py and verbose logging)

Runs pipeline steps and records per-step outputs in Mongo:
 - upload_raw_to_mongo.py    (uploads ALL raw data)
 - create_canonical_profiles.py   (limit applies here)
 - validation/run_validation.py
 - export_training_table.py
 - train_xgb.py

Writes:
 - orchestrator_runs
 - orchestrator_summary
"""
from __future__ import annotations
import argparse
import os
import sys
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Repo-root shim so running `python src/orchestrator.py` works from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PY = os.getenv("PYTHON_EXEC", sys.executable)

# pymongo safe import
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def connect_db(mongo_uri: str, mongo_db: str):
    if MongoClient is None:
        raise RuntimeError("pymongo not installed")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    return client[mongo_db]

def run_step_capture(script_relpath: str, args: list[str], repo_root: Path = REPO_ROOT, python_exec: str = PY, timeout: int | None = None) -> dict:
    """
    Run src/<script_relpath> as subprocess and capture returncode/stdout/stderr.
    Returns {"returncode", "cmd", "stdout", "stderr"}.
    """
    script_path = repo_root / "src" / script_relpath
    cmd_display = f"{python_exec} {script_path} {' '.join(shlex.quote(a) for a in args)}"
    if not script_path.exists():
        return {"returncode": 2, "cmd": cmd_display, "stdout": "", "stderr": f"script not found: {script_path}"}
    cmd = [python_exec, str(script_path)] + args
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(repo_root))
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return {"returncode": 124, "cmd": cmd_display, "stdout": out or "", "stderr": (err or "") + "\n[timeout]"}
        return {"returncode": proc.returncode, "cmd": cmd_display, "stdout": out or "", "stderr": err or ""}
    except Exception as e:
        return {"returncode": 3, "cmd": cmd_display, "stdout": "", "stderr": f"exception running subprocess: {e}"}

def insert_run_step(db, run_id: str, step: str, short_msg: str, result: dict):
    rec = {
        "run_id": run_id,
        "step": step,
        "ts": now_iso(),
        "msg": short_msg[:2000],
        "cmd": result.get("cmd"),
        "result_code": int(result.get("returncode") or 0)
    }
    try:
        db.orchestrator_runs.insert_one(rec)
    except Exception as e:
        print(f"[WARN] failed to write orchestrator_runs: {e}", file=sys.stderr)

def update_summary(db, run_id: str, key: str, result: dict):
    """
    Save detailed result under orchestrator_summary and also insert a simplified run_step row.
    """
    try:
        db.orchestrator_summary.update_one({"run_id": run_id}, {"$set": {key: result}}, upsert=True)
    except Exception as e:
        print(f"[WARN] failed to update orchestrator_summary: {e}", file=sys.stderr)
    insert_run_step(db, run_id, key, (result.get("stderr") or result.get("stdout") or "")[:1500], result)

def orchestrator_main(limit: int, applicant: str | None, llm_model: str, mongo_uri: str, mongo_db: str):
    db = connect_db(mongo_uri, mongo_db)
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    started = now_iso()
    db.orchestrator_summary.insert_one({"run_id": run_id, "applicant": applicant, "limit": limit, "llm_model": llm_model, "started_at": started})

    # ---------- upload step (all raw data) ----------
    print("[ORCH] step=upload")
    upload_args = [
        "--data", "data/synthetic_dataset",
        "--mongo_uri", mongo_uri,
        "--mongo_db", mongo_db
    ]
    res_upload = run_step_capture("upload_raw_to_mongo.py", upload_args, timeout=60*10)
    print("[ORCH] upload stdout:\n", res_upload.get("stdout") or "")
    print("[ORCH] upload stderr:\n", res_upload.get("stderr") or "")
    update_summary(db, run_id, "upload", res_upload)
    print("[ORCH] upload rc=", res_upload.get("returncode"))

    # ---------- canonical ----------
    print("[ORCH] step=canonical")
    canonical_args = [
        "--llm-model", llm_model,
        "--out", "output/canonical_profiles",
        "--local-data", "data/synthetic_dataset",
        "--mongo-uri", mongo_uri,
        "--mongo-db", mongo_db
    ]
    if applicant:
        canonical_args = ["--applicant", applicant, "--llm-model", llm_model,
                          "--out", "output/canonical_profiles",
                          "--local-data", "data/synthetic_dataset",
                          "--mongo-uri", mongo_uri, "--mongo-db", mongo_db]
    else:
        if limit and limit > 0:
            canonical_args += ["--limit", str(limit)]
    res_canonical = run_step_capture("create_canonical_profiles.py", canonical_args, timeout=60*20)
    print("[ORCH] canonical stdout:\n", res_canonical.get("stdout") or "")
    print("[ORCH] canonical stderr:\n", res_canonical.get("stderr") or "")
    update_summary(db, run_id, "canonical", res_canonical)
    print("[ORCH] canonical rc=", res_canonical.get("returncode"))

    # ---------- validation ----------
    print("[ORCH] step=validation")
    val_args = []
    if applicant:
        val_args += ["--applicant", applicant]
    else:
        if limit and limit > 0:
            val_args += ["--limit", str(limit)]
    val_args += ["--mongo-uri", mongo_uri, "--mongo-db", mongo_db]
    res_validation = run_step_capture("validation/run_validation.py", val_args, timeout=60*10)
    print("[ORCH] validation stdout:\n", res_validation.get("stdout") or "")
    print("[ORCH] validation stderr:\n", res_validation.get("stderr") or "")
    update_summary(db, run_id, "validation", res_validation)
    print("[ORCH] validation rc=", res_validation.get("returncode"))

    # ---------- export ----------
    print("[ORCH] step=export")
    export_args = ["--out", "output/training", "--mongo-uri", mongo_uri, "--mongo-db", mongo_db]
    if applicant:
        export_args += ["--applicants-file", applicant]
    else:
        if limit and limit > 0:
            export_args += ["--limit", str(limit),
                            "--split-train-size", str(max(1, (limit-1) if (limit and limit > 1) else 1))]
    res_export = run_step_capture("export_training_table.py", export_args, timeout=60*5)
    print("[ORCH] export stdout:\n", res_export.get("stdout") or "")
    print("[ORCH] export stderr:\n", res_export.get("stderr") or "")
    update_summary(db, run_id, "export", res_export)
    print("[ORCH] export rc=", res_export.get("returncode"))

    # ---------- train ----------
    print("[ORCH] step=train")
    train_args = [
        "--train", "output/training/train.csv",
        "--test", "output/training/test.csv",
        "--labels", "data/labels.csv",   # ✅ fixed to top-level labels
        "--out", "models",
        "--eval-out", "output/evaluation",
        "--seed", "42",
        "--compute-shap"
    ]
    res_train = run_step_capture("train_xgb.py", train_args, timeout=60*60)
    print("[ORCH] train stdout:\n", res_train.get("stdout") or "")
    print("[ORCH] train stderr:\n", res_train.get("stderr") or "")
    update_summary(db, run_id, "train", res_train)
    print("[ORCH] train rc=", res_train.get("returncode"))

    # ---------- finalize ----------
    finished = now_iso()
    try:
        db.orchestrator_summary.update_one(
            {"run_id": run_id},
            {"$set": {"finished_at": finished, "summary": {
                "upload": res_upload.get("returncode"),
                "canonical": res_canonical.get("returncode"),
                "validation": res_validation.get("returncode"),
                "export": res_export.get("returncode"),
                "train": res_train.get("returncode")
            }}}
        )
    except Exception as e:
        print(f"[WARN] failed to finalize summary: {e}", file=sys.stderr)

    print("[ORCH] done run_id=", run_id)
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50, help="How many canonical profiles to create / export (does NOT limit raw upload).")
    p.add_argument("--applicant", type=str, default=None, help="If set, process only this applicant (overrides limit).")
    p.add_argument("--llm-model", type=str, default=os.getenv("LLM_MODEL","mistral"))
    p.add_argument("--mongo-uri", type=str, default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    p.add_argument("--mongo-db", type=str, default=os.getenv("MONGO_DB","socialsupport"))
    args = p.parse_args()
    try:
        rc = orchestrator_main(limit=args.limit, applicant=args.applicant,
                               llm_model=args.llm_model, mongo_uri=args.mongo_uri, mongo_db=args.mongo_db)
        sys.exit(rc)
    except Exception as e:
        print(f"[FATAL] orchestrator failed: {e}", file=sys.stderr)
        try:
            if MongoClient is not None:
                db = connect_db(args.mongo_uri, args.mongo_db)
                db.orchestrator_summary.insert_one(
                    {"run_id": f"run_error_{int(time.time())}", "error": str(e), "ts": now_iso()}
                )
        except Exception:
            pass
        sys.exit(2)