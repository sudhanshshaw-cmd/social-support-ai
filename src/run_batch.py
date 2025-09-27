#!/usr/bin/env python3
"""
Validate the first N canonical profiles with Pydantic v2
and write results back to MongoDB.
"""

from pprint import pprint
from pymongo import MongoClient
from src.validation.schema_v2 import validate_and_store_v2  # directly from src/

def main(limit=10,
         mongo_uri="mongodb://localhost:27017",
         mongo_db="socialsupport"):
    client = MongoClient(mongo_uri)
    db = client[mongo_db]

    cursor = db.canonical_profiles.find({}, {"_id": 0}).limit(limit)

    for doc in cursor:
        applicant_id = doc.get("applicant_id")
        print(f"\n[INFO] Validating {applicant_id} ...")
        ok, result = validate_and_store_v2(doc, mongo_uri=mongo_uri, mongo_db=mongo_db)


        if ok:
            print(f"[SUCCESS] {applicant_id} → status={result['validation_status']}")
            # Show a clean subset for demo
            demo_view = {
                "applicant_id": result.get("applicant_id"),
                "name": result.get("name"),
                "dob": result.get("dob"),
                "normalized_income": result.get("normalized_income"),
                "credit_score": result.get("credit_score"),
                "risk_band": result.get("risk_band"),
                "validation_status": result.get("validation_status"),
                "mismatch_notes": result.get("mismatch_notes"),
            }
            pprint(demo_view)
        else:
            print(f"[FAILED] {applicant_id} → errors:")
            pprint(result)

if __name__ == "__main__":
    main(limit=10)