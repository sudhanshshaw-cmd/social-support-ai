#!/usr/bin/env python3
"""
llm_explainer_mistral.py

Generate detailed, rule-driven explanations for a candidate using a local Mistral model via Ollama.

Behavior:
- Rule-first decisioning (policy). XGBoost probability is supporting evidence used as a safeguard.
- Final decision is binary: "approve" or "reject".
- Stores results in Mongo: llm_calls (audit) and explanations.
- Dry-run path produces deterministic output (no Ollama required).

Usage:
  python src/llm_explainer_mistral.py --applicant APP100001 --dry-run
  python src/llm_explainer_mistral.py --applicant APP100001 --model mistral
"""
from __future__ import annotations
import argparse
import json
import os
import time
import uuid
import traceback
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List

# Optional / runtime imports
try:
    import ollama
except Exception:
    ollama = None

try:
    import json5
except Exception:
    json5 = None

from pymongo import MongoClient

# ---------------- CONFIG: policy thresholds (tweak as needed) ----------------
THRESHOLDS = {
    "auto_approve_prob": 0.8,         # not used to override rules, only supporting evidence
    "approve_confidence": 0.7,
    "soft_decline_prob": 0.5,
    "hard_reject_prob": 0.3,
    "mismatch_pct_threshold": 0.15,
    "low_confidence_threshold": 0.5,
    # XGB probability threshold required to accept rule-based approve
    "xgb_threshold_for_approve": 0.60
}

# ---------------- PROMPT (rule-first, few-shot + strict schema) ----------------
PROMPT_TEMPLATE = r"""
You are an expert, concise analyst that MUST produce two outputs: (A) a strict JSON object ONLY, followed by (B) a short human paragraph explanation.

INSTRUCTIONS (must be followed exactly):
1. Use ONLY the information in the provided `CONTEXT` canonical_profile and the POLICY below. Do NOT invent facts.
2. Compute a rule-based decision using the POLICY. The ML model (`xgb_prob`) is supporting evidence only — it cannot override the rules. If the ML model conflicts with the rules, mention the disagreement explicitly and recommend human review.
3. Output JSON exactly as specified below, then a single human-readable paragraph on the next line. No extra text before/after.
4. JSON must include: category, final_decision, canonical_summary, decision_score, evidence_items, short_bullets, recommended_action, xgb_prob.
5. Evidence items must reference actual fields or snippets (e.g., "declared_income: 35000", "credit_score: 515", "bank_lines: 'Salary Credit 32000'").
6. short_bullets must be a concise, ordered list of the reasoning steps (1., 2., 3.) — this is NOT hidden chain-of-thought, it's a human-facing step list.
7. decision_score is 0..1 representing rule-driven confidence (not ML probability). Use clear logic:
   - >=0.90: very strong approve
   - 0.70-0.89: approve
   - 0.40-0.69: needs_review
   - <0.40: decline/soft_decline/hard_reject depending on severity

POLICY (apply these thresholds exactly in this order; the most conservative matching outcome is selected):
- Hard-reject: if default_flag == True OR credit_score < 450 OR (debt_to_income_ratio is not None and debt_to_income_ratio > 0.7) -> category "hard_reject"
- Soft-decline (recommend enablement): if credit_score < 600 AND debt_to_income_ratio is not None AND debt_to_income_ratio > 0.35 -> "soft_decline"
- Manual review (needs_review): if mismatch_flag == True OR confidence_score < 0.7 OR (0.4 <= xgb_prob < 0.6) -> "needs_review"
- Auto-approve: if normalized_income present AND confidence_score >= 0.7 AND mismatch_flag == False AND credit_score >= 600 -> "approve"
- If multiple rules apply, choose the most conservative (hard_reject > soft_decline > needs_review > approve).

SCHEMA (JSON ONLY):
{
  "category": "<approve|needs_review|soft_decline|hard_reject>",
  "final_decision": "<approve|reject>",
  "canonical_summary": { "normalized_income": <num|null>, "income_source": "<bank|declared|llm|null>", "credit_score": <int|null>, "debt_to_income_ratio": <float|null>, "mismatch_pct": <float|null>, "confidence_score": <float|null> },
  "decision_score": <float 0-1>,
  "evidence_items": [ { "field": "<field_name>", "snippet": "<short snippet or value>", "weight": 0.0-1.0 } ],
  "short_bullets": [ "1. First reasoning step ...", "2. Second ..." ],
  "recommended_action": "<short string>",
  "xgb_prob": <float 0-1>
}
"""

# ---------------- Helper utilities ----------------
def now_iso():
    return datetime.utcnow().isoformat()

def compact_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "applicant_id", "name", "dob", "income_source", "declared_income", "bank_salary_estimate",
        "normalized_income", "confidence_score", "mismatch_flag", "mismatch_pct",
        "credit_score", "loan_count", "default_flag", "per_capita_income", "wealth_index",
        "debt_to_income_ratio", "risk_band", "llm_notes", "source_evidence"
    ]
    return {k: doc.get(k) for k in keys if k in doc}

def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "": return None
        return float(x)
    except:
        return None

def safe_int(x) -> Optional[int]:
    try:
        if x is None or x == "": return None
        return int(float(x))
    except:
        return None

# ---------------- Rule engine (deterministic) ----------------
def apply_policy_rules(profile: Dict[str, Any], xgb_prob: float) -> Tuple[str, str, float, List[Dict[str,Any]]]:
    """
    Returns (category, final_decision, decision_score, evidence_items)
      - category: rule-level category (approve|needs_review|soft_decline|hard_reject)
      - final_decision: 'approve' or 'reject' (binary)
      - decision_score: rule-driven confidence (0..1)
      - evidence_items: list of dicts {field, snippet, weight}
    """
    cat = "needs_review"
    evidence: List[Dict[str, Any]] = []
    # extract numeric fields
    norm_inc = safe_float(profile.get("normalized_income"))
    conf = safe_float(profile.get("confidence_score")) or 0.0
    mismatch_flag = bool(profile.get("mismatch_flag"))
    mismatch_pct = safe_float(profile.get("mismatch_pct"))
    credit = safe_int(profile.get("credit_score"))
    dti = safe_float(profile.get("debt_to_income_ratio"))
    default_flag = bool(profile.get("default_flag"))

    # gather evidence candidates with simple weights
    if norm_inc is not None:
        evidence.append({"field":"normalized_income","snippet":str(norm_inc),"weight":0.30})
    if profile.get("bank_salary_estimate") is not None:
        evidence.append({"field":"bank_salary_estimate","snippet":str(profile.get("bank_salary_estimate")),"weight":0.20})
    if profile.get("declared_income") is not None:
        evidence.append({"field":"declared_income","snippet":str(profile.get("declared_income")),"weight":0.10})
    if credit is not None:
        evidence.append({"field":"credit_score","snippet":str(credit),"weight":0.25})
    if dti is not None:
        evidence.append({"field":"debt_to_income_ratio","snippet":str(round(dti,3)),"weight":0.15})
    if mismatch_flag:
        evidence.append({"field":"mismatch_flag","snippet":f"mismatch_pct={mismatch_pct}","weight":0.25})
    if default_flag:
        evidence.append({"field":"default_flag","snippet":"default_flag=true","weight":0.40})

    # Decide category according to POLICY (most conservative rule wins)
    # 1) Hard reject
    if default_flag or (credit is not None and credit < 450) or (dti is not None and dti > 0.7):
        cat = "hard_reject"
        score = 0.12
    # 2) Soft-decline
    elif (credit is not None and credit < 600) and (dti is not None and dti > 0.35):
        cat = "soft_decline"
        score = 0.25
    # 3) Manual review
    elif mismatch_flag or conf < THRESHOLDS["approve_confidence"] or (0.4 <= xgb_prob < 0.6):
        cat = "needs_review"
        base = 0.45
        if conf:
            base += 0.2 * (conf - 0.5)
        if credit:
            if credit >= 700:
                base += 0.05
            elif credit < 550:
                base -= 0.05
        score = max(0.3, min(round(base,3), 0.7))
    # 4) Auto-approve
    elif norm_inc is not None and conf >= THRESHOLDS["approve_confidence"] and (not mismatch_flag) and (credit is None or credit >= 600):
        cat = "approve"
        score = 0.85
        if credit and credit >= 750:
            score = 0.93
    else:
        cat = "needs_review"
        score = 0.50

    # Final decision logic: binary
    # - Only approve when rules -> approve AND xgb_prob >= configured threshold
    xgb_thresh = THRESHOLDS.get("xgb_threshold_for_approve", 0.6)
    if cat == "approve" and (xgb_prob is not None) and (xgb_prob >= xgb_thresh):
        final = "approve"
    else:
        final = "reject"

    evidence = sorted(evidence, key=lambda e: -e.get("weight", 0))
    return cat, final, round(float(score),3), evidence

# ---------------- LLM call & parsing ----------------
def call_ollama(model_name: str, prompt: str) -> str:
    if ollama is None:
        raise RuntimeError("ollama client not installed")
    # try standard client
    try:
        resp = ollama.chat(model=model_name, messages=[{"role":"user","content": prompt}])
        if isinstance(resp, dict):
            content = resp.get("message", {}).get("content") or resp.get("content") or str(resp)
        else:
            content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
        return content
    except Exception:
        # try alternative client usage
        try:
            from ollama import Client
            client = Client()
            resp = client.chat(model=model_name, messages=[{"role":"user","content": prompt}])
            if isinstance(resp, dict):
                content = resp.get("message", {}).get("content") or resp.get("content") or str(resp)
            else:
                content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
            return content
        except Exception as e:
            raise

def parse_json_and_text(resp_text: str) -> Tuple[Optional[dict], Optional[str]]:
    if not resp_text:
        return None, None
    s = resp_text.strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        json_snip = s[first:last+1]
        try:
            parsed = json.loads(json_snip)
            rest = s[last+1:].strip()
            human = rest
            if human.startswith("\n"):
                human = human.strip()
            return parsed, human
        except Exception:
            if json5:
                try:
                    parsed = json5.loads(json_snip)
                    rest = s[last+1:].strip()
                    return parsed, rest
                except:
                    pass
    return None, s

# ---------------- Human summary builder (richer text) ----------------
def build_rich_human_text(applicant_id: str, name: Optional[str], canonical_summary: Dict[str, Any],
                          category: str, final_decision: str, decision_score: float,
                          evidence_items: List[Dict[str,Any]], xgb_prob: float) -> str:
    parts = []
    display_name = f"{name} ({applicant_id})" if name else applicant_id
    # top line: final decision + reason
    parts.append(f"Decision: LOAN {'APPROVED' if final_decision=='approve' else 'REJECTED'} (rule-level category: {category}).")
    parts.append(f"The applicant, {display_name}, was evaluated using canonical profile fields and our policy rules.")
    # income
    inc = canonical_summary.get("normalized_income")
    if inc is not None:
        try:
            parts.append(f"Normalized income is ₹{int(inc):,} per month, used for affordability checks.")
        except Exception:
            parts.append(f"Normalized income is {inc} per month.")
    # income source
    src = canonical_summary.get("income_source")
    if src:
        parts.append(f"Income source: {src}.")
    # mismatch
    mismatch = canonical_summary.get("mismatch_pct")
    if mismatch is not None and mismatch > 0:
        parts.append(f"There is an income mismatch of {round(mismatch*100,1)}% between declared and bank evidence.")
    # credit
    credit = canonical_summary.get("credit_score")
    if credit is not None:
        if credit >= 750:
            credit_desc = "excellent"
        elif credit >= 650:
            credit_desc = "good"
        elif credit >= 550:
            credit_desc = "moderate"
        else:
            credit_desc = "low and risky"
        parts.append(f"Credit score is {credit} ({credit_desc}).")
    # dti
    dti = canonical_summary.get("debt_to_income_ratio")
    if dti is not None:
        parts.append(f"Debt-to-income ratio is {round(dti,3)} (policy thresholds applied).")
    # confidence
    conf = canonical_summary.get("confidence_score")
    if conf is not None:
        parts.append(f"Profile confidence score: {round(conf,2)}.")
    # Evidence summary (top items)
    if evidence_items:
        top_snips = []
        for e in evidence_items[:3]:
            top_snips.append(f"{e['field']}={e['snippet']}")
        parts.append("Key evidence: " + "; ".join(top_snips) + ".")
    # ML note if contradicts
    ml_note = ""
    if category == "approve" and xgb_prob < THRESHOLDS.get("xgb_threshold_for_approve", 0.6):
        ml_note = f" Note: ML model probability ({xgb_prob:.2f}) is lower than our XGB safeguard threshold — recommend human review before override."
    elif category in ("hard_reject","soft_decline") and xgb_prob >= 0.7:
        ml_note = f" Note: ML model probability ({xgb_prob:.2f}) contradicts the rule-based {category} — recommend human review."
    if ml_note:
        parts.append(ml_note)
    # final sentences
    parts.append(f"Rule-driven decision score: {round(decision_score,3)}. Final action: {'APPROVE' if final_decision=='approve' else 'REJECT'} the application.")
    # recommended actions
    action_map = {
        "approve": "Auto-approve: proceed with disbursement and record decision.",
        "needs_review": "Manual review: verify documents (bank, ID) and reconcile inconsistencies before deciding.",
        "soft_decline": "Soft-decline: recommend enablement (upskilling/job placement, financial counselling).",
        "hard_reject": "Hard reject: decline support due to high financial risk or documented defaults."
    }
    parts.append(action_map.get(category, "Refer to human reviewer."))
    return " ".join(parts)

# ---------------- Main generation function ----------------
def generate_explanation_for_applicant(applicant_id: str,
                                       mongo_uri: str = "mongodb://localhost:27017",
                                       mongo_db: str = "socialsupport",
                                       model_name: str = "mistral",
                                       save_to_db: bool = True,
                                       dry_run: bool = False) -> Dict[str, Any]:
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    canonical = db.canonical_profiles.find_one({"applicant_id": applicant_id}) or {}
    if not canonical:
        raise ValueError(f"No canonical profile found for {applicant_id}")

    # fetch latest prediction if present
    pred = db.predictions.find_one({"applicant_id": applicant_id}, sort=[("created_at", -1)]) or {}
    # normalize different possible field names to xgb_prob
    xgb_prob = 0.0
    try:
        if "xgb_prob" in pred and pred.get("xgb_prob") is not None:
            xgb_prob = float(pred.get("xgb_prob") or 0.0)
        else:
            # try nested xgb_result.prob
            xgb_res = pred.get("xgb_result") or {}
            if xgb_res and xgb_res.get("prob") is not None:
                xgb_prob = float(xgb_res.get("prob") or 0.0)
            else:
                # try `probability` or `prob`
                xgb_prob = float(pred.get("prob") or pred.get("probability") or 0.0)
    except Exception:
        xgb_prob = 0.0

    top_features = pred.get("top_features") or pred.get("xgb_result", {}).get("top_features") or []

    compact = compact_profile(canonical)
    # deterministic rule-driven decision + final decision
    category_rule, final_decision, score_rule, evidence_items = apply_policy_rules(compact, xgb_prob)

    # build canonical_summary
    canonical_summary = {
        "normalized_income": compact.get("normalized_income"),
        "income_source": compact.get("income_source"),
        "credit_score": compact.get("credit_score"),
        "debt_to_income_ratio": compact.get("debt_to_income_ratio"),
        "mismatch_pct": compact.get("mismatch_pct"),
        "confidence_score": compact.get("confidence_score")
    }

    # build the payload/context for LLM (if used)
    payload = {
        "applicant_id": applicant_id,
        "canonical_profile": compact,
        "canonical_summary": canonical_summary,
        "xgb_result": {"prob": round(xgb_prob, 3), "top_features": top_features},
        "policy": THRESHOLDS,
        "rule_decision": {"category": category_rule, "decision_score": round(score_rule,3), "final_decision": final_decision}
    }

    run_id = str(uuid.uuid4())
    started = now_iso()
    llm_call_record = {
        "run_id": run_id,
        "applicant_id": applicant_id,
        "model": model_name,
        "ts": started,
        "payload_snippet": json.dumps(payload, default=str)[:4000]
    }

    # If dry-run or no ollama client, build deterministic JSON + richer human text and store
    if dry_run or ollama is None:
        # build short bullets from evidence_items (ordered)
        bullets = []
        for idx, e in enumerate(evidence_items[:6], start=1):
            bullets.append(f"{idx}. {e['field'].replace('_',' ')}: {e['snippet']}")

        recommended_action_map = {
            "approve": "Auto-approve: proceed with disbursement and record decision.",
            "needs_review": "Manual review: verify documents (bank, ID) and reconcile inconsistencies before deciding.",
            "soft_decline": "Soft-decline: recommend enablement (upskilling/job placement, financial counselling).",
            "hard_reject": "Hard reject: decline support due to high financial risk or documented defaults."
        }
        recommended_action = recommended_action_map.get(category_rule, "Refer to human reviewer.")

        json_obj = {
            "category": category_rule,
            "final_decision": final_decision,
            "canonical_summary": canonical_summary,
            "decision_score": round(score_rule,3),
            "evidence_items": evidence_items,
            "short_bullets": bullets or ["No concise evidence extracted."],
            "recommended_action": recommended_action,
            "xgb_prob": round(xgb_prob,3),
            "xgb_result": {"prob": round(xgb_prob,3), "top_features": top_features}
        }

        human = build_rich_human_text(applicant_id, canonical.get("name"), canonical_summary, category_rule, final_decision, score_rule, evidence_items, xgb_prob)

        # store llm_calls (dry-run)
        if save_to_db:
            llm_call_record.update({"dry_run": True, "response_preview": json.dumps(json_obj)[:2000], "ts_done": now_iso()})
            try:
                db.llm_calls.insert_one(llm_call_record)
            except Exception:
                pass
            try:
                db.explanations.replace_one(
                    {"applicant_id": applicant_id, "run_id": run_id},
                    {"applicant_id": applicant_id, "run_id": run_id, "json": json_obj, "human": human, "created_at": now_iso()},
                    upsert=True
                )
            except Exception:
                pass
        return {"json": json_obj, "human": human, "run_id": run_id}

    # Build prompt for LLM (we attach the full CONTEXT payload)
    prompt = PROMPT_TEMPLATE + "\n\nCONTEXT:\n" + json.dumps(payload, default=str, indent=2)

    # Call Ollama
    try:
        start = time.time()
        resp_text = call_ollama(model_name, prompt)
        latency = time.time() - start
        parsed_json, human = parse_json_and_text(resp_text)

        llm_call_record.update({"raw_response": resp_text[:4000], "latency": latency, "success": bool(parsed_json)})
        if save_to_db:
            try:
                db.llm_calls.insert_one(llm_call_record)
            except Exception:
                pass

        if parsed_json is None:
            # parsing failed: store fallback (and include raw response in human field)
            fallback = {
                "category": category_rule,
                "final_decision": final_decision,
                "canonical_summary": canonical_summary,
                "decision_score": round(score_rule,3),
                "evidence_items": evidence_items,
                "short_bullets": [f"{i+1}. {e['field']}: {e['snippet']}" for i,e in enumerate(evidence_items[:5])],
                "recommended_action": "Parse failed: refer to human reviewer",
                "xgb_prob": round(xgb_prob,3),
                "xgb_result": {"prob": round(xgb_prob,3), "top_features": top_features}
            }
            if save_to_db:
                try:
                    db.explanations.replace_one({"applicant_id": applicant_id, "run_id": run_id},
                                                {"applicant_id": applicant_id, "run_id": run_id, "json": fallback, "human": resp_text, "created_at": now_iso()},
                                                upsert=True)
                except Exception:
                    pass
            return {"json": fallback, "human": resp_text, "run_id": run_id}

        # success: ensure final_decision and xgb_prob present in parsed_json (if not, inject)
        if "final_decision" not in parsed_json:
            parsed_json["final_decision"] = final_decision
        parsed_json["xgb_prob"] = parsed_json.get("xgb_prob", round(xgb_prob,3))
        if "xgb_result" not in parsed_json:
            parsed_json["xgb_result"] = {"prob": round(xgb_prob,3), "top_features": top_features}

        if save_to_db:
            try:
                db.explanations.replace_one({"applicant_id": applicant_id, "run_id": run_id},
                                            {"applicant_id": applicant_id, "run_id": run_id, "json": parsed_json, "human": human, "created_at": now_iso()},
                                            upsert=True)
            except Exception:
                pass
        return {"json": parsed_json, "human": human, "run_id": run_id}
    except Exception as e:
        # on any error, fallback deterministic
        try:
            db.llm_calls.insert_one({**llm_call_record, "error": str(e), "ts_done": now_iso()})
        except Exception:
            pass
        return generate_explanation_for_applicant(applicant_id, mongo_uri, mongo_db, model_name, save_to_db, dry_run=True)

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant", required=True, help="Applicant ID to explain")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--mongo-db", default=os.getenv("MONGO_DB", "socialsupport"))
    parser.add_argument("--model", default="mistral", help="Ollama model name (e.g. mistral)")
    parser.add_argument("--dry-run", action="store_true", help="Don't call Ollama; use deterministic rule-based output")
    args = parser.parse_args()

    try:
        out = generate_explanation_for_applicant(args.applicant, mongo_uri=args.mongo_uri, mongo_db=args.mongo_db, model_name=args.model, save_to_db=True, dry_run=args.dry_run)
        print("=== JSON ===")
        print(json.dumps(out["json"], indent=2))
        print("\n=== HUMAN ===")
        print(out["human"])
    except Exception as exc:
        print("[ERROR] explanation failed:", str(exc))
        traceback.print_exc()
        raise