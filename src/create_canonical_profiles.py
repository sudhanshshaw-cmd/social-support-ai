#!/usr/bin/env python3
"""
create_canonical_profiles_fixed.py

Produces canonical_profiles with consistent evidence fields.
- Always supplies 'credit_json' in source_evidence (from db.raw_credit_reports || local file || {}).
- Always supplies 'bank_lines' and 'id_ocr' keys (empty lists/strings if absent).
- Optionally reprocess a single applicant with --applicant APPID.

Usage:
  source .venv/bin/activate
  python src/create_canonical_profiles_fixed.py --limit 10 --llm-model mistral --out output/canonical_profiles
  python src/create_canonical_profiles_fixed.py --applicant APP100005 --llm-model mistral --out output/canonical_profiles
"""

import argparse, json, os, re, sys, traceback
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any

# optional deps
try:
    import ollama
except Exception:
    ollama = None
try:
    import json5
except Exception:
    json5 = None
try:
    from pymongo import MongoClient
    import gridfs
    from bson import ObjectId
except Exception:
    MongoClient = None
try:
    import pandas as pd
except Exception:
    pd = None

NUM_RE = re.compile(r"(-?\d{1,3}(?:,\d{3})*|\d+)")

def now_iso(): return datetime.utcnow().isoformat()

def safe_int(x):
    if x is None or x == "": return None
    if isinstance(x, (int, float)): return int(x)
    s = re.sub(r"[^\d-]", "", str(x))
    return int(s) if s else None

def parse_json_from_model_output(s: str):
    if not s or not isinstance(s, str): return None
    s = s.strip()
    try: return json.loads(s)
    except: pass
    first = s.find('{'); last = s.rfind('}')
    if first!=-1 and last!=-1 and last>first:
        snippet = s[first:last+1]
        try: return json.loads(snippet)
        except:
            if json5:
                try: return json5.loads(snippet)
                except: pass
            try:
                repaired = snippet.replace("'", '"')
                repaired = re.sub(r",\s*}", "}", repaired)
                repaired = re.sub(r",\s*]", "]", repaired)
                return json.loads(repaired)
            except:
                if json5:
                    try: return json5.loads(repaired)
                    except: pass
    return None

def call_mistral_normalizer(llm_model: str, applicant_row: dict, extracted: dict) -> Tuple[Optional[dict], str]:
    if ollama is None: return None, "ollama_not_installed"
    instr = (
        "Return VALID JSON only with keys: name, dob (YYYY-MM-DD|null), suggested_income (number|null), "
        "source ('bank'|'id'|'declared'|'unknown'), confidence (0-1), evidence (array). No extra text."
    )
    payload = {
        "instruction": instr,
        "applicant_row": applicant_row,
        "extracted": {
            "bank_salary_estimate": extracted.get("bank_salary_estimate"),
            "salary_candidates": extracted.get("salary_candidates", [])[:6],
            "salary_lines": extracted.get("salary_lines", [])[:6],
            "id_ocr_snippet": (extracted.get("id_ocr_snippet") or "")[:800],
            "credit_report": (extracted.get("credit_report") if isinstance(extracted.get("credit_report"), dict) else None)
        }
    }
    user_prompt = "NORMALIZE_JSON_ONLY:\n" + json.dumps(payload, default=str)
    try:
        try:
            resp = ollama.chat(model=llm_model, messages=[{"role":"user","content": user_prompt}])
            if isinstance(resp, dict):
                content = resp.get("message",{}).get("content") or resp.get("content") or str(resp)
            else:
                content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
        except Exception:
            from ollama import Client
            client = Client()
            resp = client.chat(model=llm_model, messages=[{"role":"user","content": user_prompt}])
            if isinstance(resp, dict):
                content = resp.get("message",{}).get("content") or resp.get("content") or str(resp)
            else:
                content = getattr(resp, "message").content if getattr(resp, "message", None) else str(resp)
        return parse_json_from_model_output(content), content
    except Exception as e:
        return None, f"llm_exception:{e}"

def fetch_credit_json(db, applicant_id: str, local_data_folder: str):
    """Try DB.raw_credit_reports -> local file fallback -> return {}"""
    # 1) DB
    try:
        doc = db.raw_credit_reports.find_one({"applicant_id": applicant_id})
        if doc and doc.get("credit_report"):
            return doc.get("credit_report")
    except Exception:
        pass
    # 2) local file fallback
    try:
        p = Path(local_data_folder) / "credit_reports" / f"{applicant_id}_credit.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

def ensure_extracted_defaults(extracted: dict):
    if extracted is None:
        extracted = {}
    # ensure keys exist and predictable types
    extracted.setdefault("bank_salary_estimate", None)
    extracted.setdefault("salary_lines", [])
    extracted.setdefault("salary_candidates", [])
    extracted.setdefault("id_ocr_snippet", "")
    extracted.setdefault("bank_text_snippet", "")
    extracted.setdefault("credit_report", {})
    extracted.setdefault("confidence", {"bank_salary_estimate": 0.25})
    return extracted

# lightweight resolve income (same deterministic rules)
def resolve_income(app_row: dict, extracted: dict, llm: dict):
    declared = safe_int(app_row.get("monthly_income") if app_row else None) or safe_int(extracted.get("declared_income"))
    bank_est = extracted.get("bank_salary_estimate")
    bank_conf = float(extracted.get("confidence",{}).get("bank_salary_estimate") or 0.25)
    try:
        if bank_est is not None:
            bank_est = int(bank_est)
    except:
        bank_est = safe_int(bank_est)
    llm_income = None; llm_conf = 0.0
    if llm:
        llm_income = llm.get("suggested_income")
        try:
            llm_income = int(llm_income) if llm_income is not None else None
        except:
            llm_income = None
        llm_conf = float(llm.get("confidence") or 0)
    chosen=None; source=None
    if llm_income and llm_conf >= 0.85:
        chosen=llm_income; source="llm"
    if not chosen and bank_est and bank_conf>=0.6:
        chosen=bank_est; source="bank"
    if not chosen and declared and declared>0:
        chosen=declared; source="declared"
    mismatch_pct=None
    if declared and chosen:
        try: mismatch_pct=round(abs(declared-chosen)/float(declared),3)
        except: mismatch_pct=None
    return {
        "normalized_income": chosen,
        "income_source": source,
        "bank_confidence": round(bank_conf or 0,3),
        "llm_confidence": round(llm_conf or 0,3),
        "declared_income": declared,
        "mismatch_pct": mismatch_pct
    }

def compute_derived(profile: Dict[str,Any]):
    normalized = profile.get("normalized_income")
    total_assets = profile.get("total_assets")
    total_liabilities = profile.get("total_liabilities")
    credit_score = profile.get("credit_score")
    per_capita = None
    if normalized: per_capita = round(float(normalized)/1.0,2)
    wealth_index = None
    if total_assets is not None or total_liabilities is not None:
        a = total_assets or 0; l = total_liabilities or 0
        wealth_index = float(a) - float(l)
    dti = None
    if normalized and total_liabilities is not None:
        try: dti = round(float(total_liabilities)/float(max(1,normalized)),3)
        except: dti = None
    risk_band = "unknown"
    if credit_score is not None:
        try:
            cs = int(credit_score)
            risk_band = "low" if cs>=700 else ("medium" if cs>=550 else "high")
        except: risk_band = "unknown"
    return per_capita, wealth_index, dti, risk_band

def generate_summary(profile: Dict[str,Any]) -> str:
    name = profile.get("name") or profile.get("applicant_id")
    dob = profile.get("dob")
    age=None
    try:
        if dob and isinstance(dob,str):
            d = datetime.fromisoformat(dob).date(); today = date.today()
            age = today.year - d.year - ((today.month,today.day) < (d.month,d.day))
    except:
        age=None
    age_txt = f", {age} yrs" if age is not None else ""
    decl = profile.get("declared_income"); bank = profile.get("bank_salary_estimate"); norm = profile.get("normalized_income")
    money_parts = []
    if decl: money_parts.append(f"Declared ₹{decl}")
    if bank: money_parts.append(f"Bank ₹{bank}")
    if norm and norm not in (bank,decl): money_parts.append(f"Normalized ₹{norm}")
    money = ", ".join(money_parts) if money_parts else "No income evidence"
    cs = profile.get("credit_score")
    risk = profile.get("risk_band") or "unknown"
    return f"{name}{age_txt}. {money}. Credit score {cs if cs else 'N/A'}. Risk {risk}."

def run(args):
    if MongoClient is None:
        print("[ERROR] pymongo required", file=sys.stderr); sys.exit(1)
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    try: client.admin.command('ping')
    except Exception as e:
        print(f"[ERROR] cannot connect to Mongo: {e}", file=sys.stderr); sys.exit(1)
    db = client[args.mongo_db]
    source = db[args.source_collection]
    canonical = db["canonical_profiles"]

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # optional CSV fallback table load
    applicant_table = {}
    if pd is not None and Path(args.local_data).exists():
        csvp = Path(args.local_data) / "applicants.csv"
        xlsxp = Path(args.local_data) / "applicants.xlsx"
        try:
            if csvp.exists():
                df = pd.read_csv(csvp, dtype=str).fillna("")
            elif xlsxp.exists():
                df = pd.read_excel(xlsxp, dtype=str).fillna("")
            else:
                df = None
            if df is not None:
                for _, r in df.iterrows():
                    aid = str(r.get("applicant_id") or r.get("id") or "").strip()
                    if aid:
                        applicant_table[aid] = r.to_dict()
        except Exception:
            applicant_table = {}

    # build query: either single applicant or cursor with limit
    if args.applicant:
        cursor = [ source.find_one({"applicant_id": args.applicant}) or db.raw_bundles.find_one({"applicant_id": args.applicant}) or source.find_one({"applicant_id": args.applicant}) ]
    else:
        proj = {"_id":1, "applicant_id":1, "applicant_row":1, "file_ids":1}
        cursor = source.find({}, projection=proj)
        if args.limit and args.limit>0:
            cursor = cursor.limit(args.limit)

    processed = 0
    for doc in cursor:
        try:
            if not doc: continue
            applicant_id = (doc.get("applicant_id") or (doc.get("applicant_row") or {}).get("applicant_id"))
            if not applicant_id:
                print("[WARN] skipping doc without applicant_id"); continue
            app_row = doc.get("applicant_row") or {}
            if (not app_row or len(app_row)==0) and applicant_table.get(str(applicant_id)):
                app_row = applicant_table.get(str(applicant_id))

            # prefer existing raw_bundles extracted if present
            raw_bundle = db.raw_bundles.find_one({"applicant_id": applicant_id})
            if raw_bundle:
                extracted = ensure_extracted_defaults(raw_bundle.get("extracted", {}))
                llm_parsed = raw_bundle.get("llm_normalized") or {}
                llm_raw_text = raw_bundle.get("llm_raw") or ""
                id_ocr = (raw_bundle.get("raw_text") or {}).get("id_ocr_snippet","") or extracted.get("id_ocr_snippet","")
            else:
                # try source doc's extracted if present
                extracted = ensure_extracted_defaults(doc.get("extracted") or {})
                llm_parsed = doc.get("llm_normalized") or {}
                llm_raw_text = doc.get("llm_raw") or ""
                id_ocr = extracted.get("id_ocr_snippet","")

            # ENFORCE: fetch credit_json uniformly
            credit_json = fetch_credit_json(db, applicant_id, args.local_data)
            extracted["credit_report"] = credit_json or extracted.get("credit_report") or {}

            # try to get name/dob from app_row, else from llm_parsed (if confident), else from id_ocr text
            name = app_row.get("name") or app_row.get("full_name") or (llm_parsed.get("name") if llm_parsed and (llm_parsed.get("confidence") or 0)>=0.7 else None)
            dob = app_row.get("dob") or (llm_parsed.get("dob") if llm_parsed and (llm_parsed.get("confidence") or 0)>=0.7 else None)

            # minimal demographic defaults - ensure keys exist
            family_size = safe_int(app_row.get("family_size")) if app_row else None
            gender = app_row.get("gender") if app_row else None

            # if still missing, weak parse id_ocr
            if (not name or not dob) and id_ocr:
                m = re.search(r"Name[:\s\-]*([A-Za-z .']{2,80})", id_ocr, re.I)
                if m and not name: name = m.group(1).strip()
                m2 = re.search(r"(\d{4}-\d{2}-\d{2})", id_ocr)
                if m2 and not dob: dob = m2.group(1)
                m3 = re.search(r"\b(M|F|Male|Female)\b", id_ocr, re.I)
                if m3 and not gender: gender = "F" if m3.group(1).lower().startswith("f") else "M"

            # ensure extracted has consistent keys
            extracted = ensure_extracted_defaults(extracted)
            # update credit_report consistently
            extracted["credit_report"] = extracted.get("credit_report") or credit_json or {}

            # call LLM if available for additional normalization (best-effort)
            llm_parsed_runtime = None; llm_raw_runtime = None
            try:
                parsed, raw = call_mistral_normalizer(args.llm_model, app_row, extracted)
                llm_parsed_runtime, llm_raw_runtime = parsed, raw
                if parsed:
                    # supplement name/dob only if missing and LLM confident
                    try:
                        if not name and parsed.get("name") and (parsed.get("confidence") or 0)>=0.7:
                            name = parsed.get("name")
                        if not dob and parsed.get("dob") and (parsed.get("confidence") or 0)>=0.7:
                            dob = parsed.get("dob")
                    except: pass
                    # merge parsed into llm_parsed
                    llm_parsed = {**(llm_parsed or {}), **parsed}
                if raw:
                    llm_raw_text = raw
            except Exception:
                pass

            # income resolution
            income_info = resolve_income(app_row, extracted, llm_parsed or {})

            canonical_doc = {
                "applicant_id": applicant_id,
                "name": name,
                "dob": dob,
                "gender": gender,
                "family_size": family_size,
                "declared_income": income_info.get("declared_income"),
                "bank_salary_estimate": extracted.get("bank_salary_estimate"),
                "normalized_income": income_info.get("normalized_income"),
                "income_source": income_info.get("income_source"),
                "bank_confidence": income_info.get("bank_confidence"),
                "llm_confidence": income_info.get("llm_confidence"),
                "avg_monthly_expense": safe_int(app_row.get("avg_monthly_expense") if app_row else None) or 0,
                "total_assets": safe_int(app_row.get("total_assets") if app_row else None),
                "total_liabilities": safe_int(app_row.get("total_liabilities") or extracted.get("credit_report",{}).get("total_debt")),
                "debt_to_income_ratio": None,
                "credit_score": safe_int(extracted.get("credit_report",{}).get("credit_score")),
                "loan_count": safe_int(extracted.get("credit_report",{}).get("outstanding_loans")),
                "default_flag": bool((extracted.get("credit_report",{}) or {}).get("remarks") and "default" in str((extracted.get("credit_report",{}) or {}).get("remarks")).lower()),
                "mismatch_pct": income_info.get("mismatch_pct"),
                # evidence ALWAYS present and predictable shape
                "source_evidence": {
                    "bank_lines": extracted.get("salary_lines") or [],
                    "id_ocr": (extracted.get("id_ocr_snippet") or "")[:2000],
                    "credit_json": extracted.get("credit_report") or {}
                },
                "llm_notes": (llm_parsed.get("evidence") if isinstance(llm_parsed, dict) else None),
                "llm_raw": (llm_raw_runtime or "")[:4000],
                "created_at": now_iso(),
                "updated_at": now_iso()
            }

            # derived fields
            per_capita, wealth_index, dti, risk_band = compute_derived(canonical_doc)
            canonical_doc["per_capita_income"] = per_capita
            canonical_doc["wealth_index"] = wealth_index
            canonical_doc["debt_to_income_ratio"] = dti
            canonical_doc["risk_band"] = risk_band

            # confidence/mismatch flags (simple deterministic)
            canonical_doc["mismatch_flag"] = True if (canonical_doc.get("mismatch_pct") is not None and canonical_doc.get("mismatch_pct")>0.15) else False

            # simple confidence aggregator (base)
            base = 0.2
            if canonical_doc.get("income_source") == "bank": base += 0.4 * float(canonical_doc.get("bank_confidence") or 0)
            if canonical_doc.get("income_source") == "llm": base += 0.5 * float(canonical_doc.get("llm_confidence") or 0)
            cs = canonical_doc.get("credit_score")
            if cs is not None:
                try:
                    cs_i = int(cs)
                    if cs_i >= 700: base += 0.05
                    elif cs_i >= 650: base += 0.02
                    elif cs_i < 550: base -= 0.05
                except: pass

            # ---- DEMO: add deterministic boosts for clearer 'ok' cases ----
            # Boost confidence when there is clear evidence:
            # - if normalized income present (explicitly derived), boost
            # - if credit score is strong, boost more
            boost = 0.0
            if canonical_doc.get("normalized_income"):
                # evidence of income from bank/llm/declared
                boost += 0.15
            if cs is not None:
                try:
                    if cs_i >= 750:
                        boost += 0.20
                    elif cs_i >= 700:
                        boost += 0.12
                    elif cs_i >= 650:
                        boost += 0.06
                except:
                    pass
            # Reduce boost a bit if mismatch is high
            mp = canonical_doc.get("mismatch_pct") or 0.0
            if mp > 0.20:
                boost -= 0.10

            final_score = min(max(base + boost, 0.0), 0.99)
            canonical_doc["confidence_score"] = round(final_score, 3)

            # Demo-friendly validation rule
            mismatch_val = canonical_doc.get("mismatch_pct") or 0.0
            if canonical_doc["confidence_score"] >= 0.6 and mismatch_val <= 0.15:
                canonical_doc["validation_status"] = "ok"
            else:
                canonical_doc["validation_status"] = "needs_review"

            canonical_doc["profile_summary"] = generate_summary(canonical_doc)

            # write to Mongo
            canonical.replace_one({"applicant_id": applicant_id}, canonical_doc, upsert=True)
            # optional local write
            if out_dir:
                with open(out_dir / f"{applicant_id}.json", "w", encoding="utf-8") as fh:
                    json.dump(canonical_doc, fh, indent=2, ensure_ascii=False)

            processed += 1
            print(f"[OK] canonical written for {applicant_id} (credit_json_present={bool(canonical_doc['source_evidence']['credit_json'])})")
        except Exception as e:
            print(f"[ERROR] failed {doc.get('_id') if doc else 'doc'}: {e}", file=sys.stderr)
            traceback.print_exc()
    print(f"Done. Processed {processed} applicants.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    p.add_argument("--mongo-db", default=os.getenv("MONGO_DB","socialsupport"))
    p.add_argument("--source-collection", default="raw_applicants")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--applicant", type=str, default=None, help="Process only this applicant_id")
    p.add_argument("--llm-model", default="mistral")
    p.add_argument("--out", default="output/canonical_profiles")
    p.add_argument("--local-data", default="data/synthetic_dataset", help="fallback folder for credit_reports etc.")
    args = p.parse_args()
    if args.out == "": args.out = None
    run(args)



    