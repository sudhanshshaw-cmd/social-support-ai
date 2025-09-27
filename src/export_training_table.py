#!/usr/bin/env python3
"""
export_training_table.py (updated)

Exports a flat CSV features table from canonical_profiles stored in Mongo.

Behavior updates:
- Keeps 'applicant_id' in the main CSVs for traceability.
- ALSO writes feature-only CSVs (train_features.csv / test_features.csv) which
  exclude 'applicant_id' and datetime columns so they can be used directly for ML
  without risk of accidental leakage.
- Writes feature_cols.json listing ordered feature column names.

Usage examples:
  python src/export_training_table.py --mongo-uri "$MONGO_URI" --mongo-db "$MONGO_DB" --limit 50 --out output/training --split-train-size 40
  python src/export_training_table.py --mongo-uri "$MONGO_URI" --mongo-db "$MONGO_DB" --applicants-file data/test_ids.txt --out output/training --split-train-size 40

Requirements:
  pip install pandas pymongo python-dateutil
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime, date
from dateutil import parser as date_parser

import pandas as pd
from pymongo import MongoClient

# Features we care about for modelling (order matters)
DEFAULT_NUMERIC = [
    "age",
    "declared_income",
    "bank_salary_estimate",
    "normalized_income",
    "avg_monthly_expense",
    "total_assets",
    "total_liabilities",
    "per_capita_income",
    "wealth_index",
    "debt_to_income_ratio",
    "confidence_score",
    "mismatch_pct",
    "credit_score",
    "loan_count",
    "family_size",
]

DEFAULT_CATEGORICAL = [
    "income_source",
    "risk_band",
    "gender",
]

# Combined ordered feature list (numeric first, then categorical)
DEFAULT_FEATURES = DEFAULT_NUMERIC + DEFAULT_CATEGORICAL

def safe_get(d, key, default=None):
    v = d.get(key, default)
    if v is None:
        return default
    return v

def parse_date_to_age(dob_str):
    if not dob_str:
        return None
    try:
        if isinstance(dob_str, (date, datetime)):
            dob = dob_str if isinstance(dob_str, date) else dob_str.date()
        else:
            dob = date_parser.parse(str(dob_str)).date()
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except Exception:
        return None

def canonical_cursor(db, source_collection="canonical_profiles", limit: int = 0):
    coll = db[source_collection]
    if limit and limit > 0:
        return coll.find({}, {}).sort("created_at", 1).limit(limit)
    return coll.find({}).sort("created_at", 1)

def document_to_row(doc, features):
    """
    Map a canonical_profiles document to a flat dict row containing the requested features.
    Will coerce types and add derived 'age'.
    """
    out = {}
    # applicant_id and metadata
    out["applicant_id"] = doc.get("applicant_id")
    out["created_at"] = doc.get("created_at")
    out["updated_at"] = doc.get("updated_at")
    out["dob"] = doc.get("dob")

    # ensure canonical fields exist (fallbacks)
    # direct numeric/categorical copy if present
    for f in features:
        if f in ("age",):
            continue
        out[f] = doc.get(f) if doc.get(f) is not None else None

    # derived / fallback for normalized_income
    if out.get("normalized_income") in (None, "", 0):
        out["normalized_income"] = doc.get("normalized_income") or doc.get("declared_income") or doc.get("bank_salary_estimate") or None

    # age from dob
    out["age"] = parse_date_to_age(doc.get("dob") or out.get("dob"))

    return out

def export(args):
    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[args.mongo_db]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # determine docs to export
    if args.applicants_file:
        with open(args.applicants_file, "r", encoding="utf-8") as fh:
            applicants = [x.strip() for x in fh.read().splitlines() if x.strip()]
        docs = []
        for aid in applicants:
            d = db.canonical_profiles.find_one({"applicant_id": aid})
            if d:
                docs.append(d)
    else:
        docs = list(canonical_cursor(db, limit=args.limit))

    rows = []
    for doc in docs:
        row = document_to_row(doc, DEFAULT_FEATURES)
        rows.append(row)

    if not rows:
        print("[WARN] No canonical profiles found with the given parameters.")
        return

    df = pd.DataFrame(rows)

    # numeric conversion for recommended numeric columns (including age)
    numeric_columns = [c for c in DEFAULT_NUMERIC + ["age"] if c in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Re-order columns - keep applicant_id and metadata first, then features
    metadata_cols = ["applicant_id", "dob", "created_at", "updated_at"]
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    ordered_cols = metadata_cols + feature_cols
    # ensure uniqueness and presence
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    df = df[ordered_cols]

    csv_path = out_dir / "training_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] training table written to {csv_path}  (rows={len(df)})")

    # If splitting into train/test, write both the full CSVs (with applicant_id) and feature-only CSVs
    if args.split_train_size:
        n_train = args.split_train_size
        n_total = len(df)
        if n_train >= n_total:
            print(f"[WARN] requested train size >= total rows ({n_train} >= {n_total}) — skipping split")
        else:
            train_df = df.iloc[:n_train].copy()
            test_df = df.iloc[n_train:].copy()

            train_path = out_dir / "train.csv"
            test_path = out_dir / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"[OK] train/test split written: {train_path} ({len(train_df)})  {test_path} ({len(test_df)})")

            # create feature-only DataFrames (drop metadata columns)
            drop_cols = ["applicant_id", "created_at", "updated_at", "dob"]
            feature_cols_ordered = [c for c in feature_cols if c not in drop_cols]
            train_feat = train_df[feature_cols_ordered].copy()
            test_feat = test_df[feature_cols_ordered].copy()

            train_feat_path = out_dir / "train_features.csv"
            test_feat_path = out_dir / "test_features.csv"
            train_feat.to_csv(train_feat_path, index=False)
            test_feat.to_csv(test_feat_path, index=False)
            print(f"[OK] feature-only CSVs written: {train_feat_path} ({len(train_feat)})  {test_feat_path} ({len(test_feat)})")

            # write feature cols JSON
            feature_cols_json = out_dir / "feature_cols.json"
            with open(feature_cols_json, "w", encoding="utf-8") as fh:
                json.dump(feature_cols_ordered, fh, indent=2)
            print(f"[OK] feature column list written to {feature_cols_json}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    p.add_argument("--mongo-db", default=os.getenv("MONGO_DB","socialsupport"))
    p.add_argument("--limit", type=int, default=50, help="If no applicants_file provided, get up to this many canonical profiles ordered by created_at")
    p.add_argument("--applicants-file", type=str, default=None, help="Path to newline-separated applicant IDs to export (overrides --limit)")
    p.add_argument("--out", type=str, default="output/training", help="Output folder")
    p.add_argument("--split-train-size", type=int, default=40, dest="split_train_size", help="If set, writes train.csv (first N) and test.csv (remaining)")
    args = p.parse_args()
    export(args)