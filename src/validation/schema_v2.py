from __future__ import annotations
# src/validation/schema_v2.py

from typing import Optional, List, Any, Dict
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from pymongo import MongoClient

SCHEMA_VERSION = "2.0"

class CreditJSON(BaseModel):
    applicant_id: Optional[str] = None
    credit_score: Optional[int] = None
    outstanding_loans: Optional[int] = None
    total_debt: Optional[int] = None
    remarks: Optional[str] = None

class SourceEvidence(BaseModel):
    bank_lines: List[str] = Field(default_factory=list)
    id_ocr: str = ""
    credit_json: CreditJSON = Field(default_factory=CreditJSON)

class CanonicalProfile(BaseModel):
    applicant_id: str
    name: Optional[str] = None
    dob: Optional[date] = None

    declared_income: Optional[int] = None
    bank_salary_estimate: Optional[int] = None
    normalized_income: Optional[int] = None
    income_source: Optional[str] = None
    bank_confidence: float = 0.0
    llm_confidence: float = 0.0

    avg_monthly_expense: Optional[int] = 0
    total_assets: Optional[int] = None
    total_liabilities: Optional[int] = None
    debt_to_income_ratio: Optional[float] = None

    credit_score: Optional[int] = None
    loan_count: Optional[int] = None
    default_flag: bool = False

    mismatch_pct: Optional[float] = None
    mismatch_flag: bool = False
    mismatch_notes: List[str] = Field(default_factory=list)

    confidence_score: float = 0.0
    validation_status: Optional[str] = None

    source_evidence: SourceEvidence = Field(default_factory=SourceEvidence)
    llm_notes: Optional[Any] = None
    profile_summary: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    schema_version: Optional[str] = None

    # -- field validators (coerce ints/dates) --
    @field_validator(
        "declared_income",
        "bank_salary_estimate",
        "normalized_income",
        "avg_monthly_expense",
        "total_assets",
        "total_liabilities",
        "credit_score",
        "loan_count",
        mode="before"
    )
    def _coerce_ints(cls, v):
        if v in (None, ""):
            return None
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        s = "".join(ch for ch in s if ch.isdigit() or ch == "-")
        return int(s) if s else None

    @field_validator("dob", mode="before")
    def _parse_dob(cls, v):
        if v in (None, ""):
            return None
        if isinstance(v, date):
            return v
        s = str(v).strip()
        # accept YYYY-MM-DD or D/M/YYYY
        try:
            return date.fromisoformat(s)
        except Exception:
            # try DD/MM/YYYY
            parts = s.replace("-", "/").split("/")
            if len(parts) == 3:
                d, m, y = parts
                try:
                    return date(int(y), int(m), int(d))
                except:
                    pass
        raise ValueError("dob must be YYYY-MM-DD or null")

    @model_validator(mode="after")
    def validate_ranges(self):
        # credit score sanity
        if self.credit_score is not None:
            if not (0 <= int(self.credit_score) <= 1000):
                raise ValueError("credit_score must be between 0 and 1000")
        # mismatch_pct sanity
        if self.mismatch_pct is not None:
            if not (0.0 <= float(self.mismatch_pct) <= 10.0):
                raise ValueError("mismatch_pct looks invalid")
        return self

# Ensure Pydantic v2 fully initializes models when using postponed annotations.
# Calling model_rebuild() fixes the "not fully defined" error in some environments.
# Wrap in try/except to be safe for older/newer pydantic versions.
for _cls in (CreditJSON, SourceEvidence, CanonicalProfile):
    try:
        _cls.model_rebuild()
    except Exception:
        try:
            # older aliases / internal names
            getattr(_cls, "model_rebuild", lambda: None)()
        except Exception:
            pass

def _serialize_dates(obj):
    """
    Recursively convert date / datetime objects in dict/list to ISO strings.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_dates(v) for v in obj]
    # datetime/date -> ISO string
    import datetime as _dt
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        # convert date -> ISO date string (YYYY-MM-DD)
        return obj.isoformat()
    return obj

def validate_and_store_v2(profile_doc: dict,
                          mongo_uri: str = "mongodb://localhost:27017",
                          mongo_db: str = "socialsupport",
                          canonical_collection: str = "canonical_profiles",
                          validated_collection: str = "canonical_profiles_validated",
                          validator_name: str = "pydantic_v2"):
    """
    Pydantic v2 validation helper with safe serialization of date/datetime
    for MongoDB storage (avoids bson encoding errors).

    This variant ALWAYS recomputes validation_status (validator is authoritative)
    and sets 'updated_at' to the current time before upsert.
    """
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    canonical_coll = db[canonical_collection]
    validated_coll = db[validated_collection]

    ts = datetime.utcnow()
    doc = dict(profile_doc)
    doc["schema_version"] = SCHEMA_VERSION
    try:
        model = CanonicalProfile.model_validate(doc)  # v2 API
        # plain Python types (may include date/datetime)
        v = model.model_dump()

        # ---- ALWAYS recompute validation_status from up-to-date fields ----
        score = float(v.get("confidence_score", 0.0) or 0.0)
        mismatch = float(v.get("mismatch_pct") or 0.0)

        # DEMO rule (adjust as needed): ok if score >= 0.4 and mismatch <= 0.15
        if score >= 0.4 and mismatch <= 0.15:
            v["validation_status"] = "ok"
        else:
            v["validation_status"] = "needs_review"

        # update validator metadata and timestamps
        v["validated_by"] = validator_name
        v["validated_at"] = ts.isoformat()
        # ensure canonical document has updated_at and created_at (create if missing)
        v["updated_at"] = ts.isoformat()
        if not v.get("created_at"):
            v["created_at"] = ts.isoformat()
        v["schema_version"] = SCHEMA_VERSION

        # serialize dates/datetimes to ISO strings for Mongo
        serializable_doc = _serialize_dates(v)

        # upsert canonical (store validated representation)
        canonical_coll.replace_one({"applicant_id": serializable_doc["applicant_id"]}, serializable_doc, upsert=True)

        # write validation run (also serialized)
        validated_record = {
            "applicant_id": serializable_doc["applicant_id"],
            "schema_version": SCHEMA_VERSION,
            "validation_status": serializable_doc.get("validation_status"),
            "validation_errors": [],
            "validated_by": validator_name,
            "validated_at": ts.isoformat(),
            "doc": serializable_doc
        }
        validated_coll.insert_one(validated_record)
        return True, serializable_doc
    except ValidationError as e:
        # build structured error list
        errors = e.errors()
        # serialize original doc for storage (make dates safe)
        safe_original = _serialize_dates(doc)
        validated_coll.insert_one({
            "applicant_id": safe_original.get("applicant_id"),
            "schema_version": SCHEMA_VERSION,
            "validation_status": "invalid",
            "validation_errors": errors,
            "validated_by": validator_name,
            "validated_at": ts.isoformat(),
            "doc": safe_original
        })
        safe_original["validation_status"] = "invalid"
        safe_original["updated_at"] = ts.isoformat()
        canonical_coll.replace_one({"applicant_id": safe_original.get("applicant_id")}, safe_original, upsert=True)
        return False, {"validation_errors": errors}