#!/usr/bin/env python3
"""
upload_raw_to_mongo_simple.py

Minimal script to upload raw dataset into MongoDB.

- Inserts each row from data/synthetic_dataset/applicants.csv into collection `raw_applicants`.
- Stores related files (bank PDFs/CSVs, ID images) into GridFS, saving their file IDs in the applicant document.
- Stores credit JSONs into `raw_credit_reports`.

Usage:
  python src/upload_raw_to_mongo_simple.py --data data/synthetic_dataset --limit 3 --mongo_uri "$MONGO_URI" --mongo_db "$MONGO_DB"
"""
import argparse, os, sys, json
from pathlib import Path
from pymongo import MongoClient
import gridfs
import pandas as pd

def connect_mongo(uri, dbname):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[dbname]
    fs = gridfs.GridFS(db)
    return client, db, fs

def upload_binary(fs, filepath: Path, applicant_id: str, file_type: str):
    with open(filepath, "rb") as fh:
        meta = {"applicant_id": applicant_id, "filename": filepath.name, "file_type": file_type}
        return fs.put(fh, filename=filepath.name, metadata=meta)

def run(data_dir: Path, mongo_uri: str, mongo_db: str, limit: int = 0):
    client, db, fs = connect_mongo(mongo_uri, mongo_db)
    print(f"[MONGO] Connected to {mongo_uri} DB={mongo_db}")

    # Load applicants
    applicants_csv = data_dir / "applicants.csv"
    df = pd.read_csv(applicants_csv)
    if limit and limit > 0:
        df = df.head(limit)

    applicants_coll = db["raw_applicants"]
    credit_coll = db["raw_credit_reports"]

    for _, row in df.iterrows():
        applicant_id = str(row["applicant_id"])
        doc = row.dropna().to_dict()
        file_ids = {}

        # Bank file
        bank_dir = data_dir / "bank_statements"
        for ext in [".pdf", ".csv"]:
            bank_file = bank_dir / f"{applicant_id}_bank{ext}"
            if bank_file.exists():
                file_ids["bank"] = str(upload_binary(fs, bank_file, applicant_id, "bank"))
                break

        # ID file
        ids_dir = data_dir / "ids"
        for ext in [".png", ".jpg", ".jpeg"]:
            id_file = ids_dir / f"{applicant_id}_id{ext}"
            if id_file.exists():
                file_ids["id"] = str(upload_binary(fs, id_file, applicant_id, "id"))
                break

        # Credit report
        credit_dir = data_dir / "credit_reports"
        credit_file = credit_dir / f"{applicant_id}_credit.json"
        if credit_file.exists():
            credit_data = json.load(open(credit_file, "r", encoding="utf-8"))
            credit_coll.replace_one({"applicant_id": applicant_id}, {"applicant_id": applicant_id, "credit_report": credit_data}, upsert=True)

        doc["file_ids"] = file_ids
        applicants_coll.replace_one({"applicant_id": applicant_id}, doc, upsert=True)
        print(f"[OK] Inserted {applicant_id} with files {list(file_ids.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_dataset")
    parser.add_argument("--mongo_uri", type=str, default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    parser.add_argument("--mongo_db", type=str, default=os.getenv("MONGO_DB","socialsupport"))
    parser.add_argument("--limit", type=int, default=0, help="0 = all applicants")
    args = parser.parse_args()
    run(Path(args.data), args.mongo_uri, args.mongo_db, args.limit)