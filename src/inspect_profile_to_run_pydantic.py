#!/usr/bin/env python3
"""
inspect_profile.py

Usage:
  python src/inspect_profile.py APP100000
"""

import sys
import json
from pymongo import MongoClient
import os

def main(applicant_id: str):
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "socialsupport")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]

    doc = db.canonical_profiles.find_one({"applicant_id": applicant_id}, {"_id": 0})
    if not doc:
        print(f"[ERROR] No canonical profile found for {applicant_id}")
        return

    print(json.dumps(doc, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inspect_profile.py <APPLICANT_ID>")
        sys.exit(1)
    main(sys.argv[1])