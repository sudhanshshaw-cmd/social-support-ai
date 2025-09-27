# src/validation/run_validation.py
#!/usr/bin/env python3
"""
Run pydantic v2 validation over canonical_profiles using schema_v2.validate_and_store_v2.

Usage:
  python src/validation/run_validation.py --limit 50
  python src/validation/run_validation.py --applicants-file data/test_ids.txt
"""

import argparse, os, sys, json
from pymongo import MongoClient
from datetime import datetime
# locate schema_v2 (assumes file at src/validation/schema_v2.py)
here = os.path.dirname(__file__)
schema_path = os.path.join(here, "schema_v2.py")
# import the module by path
import importlib.util
spec = importlib.util.spec_from_file_location("schema_v2", schema_path)
schema_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema_mod)
validate_and_store_v2 = schema_mod.validate_and_store_v2

def get_db(uri, dbname):
    client = MongoClient(uri)
    return client[dbname]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI","mongodb://localhost:27017"))
    p.add_argument("--mongo-db", default=os.getenv("MONGO_DB","socialsupport"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--applicants-file", type=str, default=None)
    p.add_argument("--canonical-collection", type=str, default="canonical_profiles")
    args = p.parse_args()

    db = get_db(args.mongo_uri, args.mongo_db)
    coll = db[args.canonical_collection]

    if args.applicants_file:
        with open(args.applicants_file, "r", encoding="utf-8") as fh:
            applicants = [x.strip() for x in fh.read().splitlines() if x.strip()]
        docs = [coll.find_one({"applicant_id": a}) for a in applicants]
        docs = [d for d in docs if d]
    else:
        cursor = coll.find({}).sort("created_at", 1)
        if args.limit and args.limit>0:
            cursor = cursor.limit(args.limit)
        docs = list(cursor)

    total = 0
    success = 0
    for d in docs:
        total += 1
        ok, res = validate_and_store_v2(d, mongo_uri=args.mongo_uri, mongo_db=args.mongo_db)
        if ok:
            success += 1
            print(f"[OK] validated {d.get('applicant_id')}")
        else:
            print(f"[ERR] {d.get('applicant_id')} -> {res.get('validation_errors')}")
    print(f"Done. validated {success}/{total}")

if __name__ == "__main__":
    main()