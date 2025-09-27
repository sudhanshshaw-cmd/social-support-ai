# src/agents/tools_wrapper.py
import os, sys, subprocess, traceback, json
from typing import Optional, Dict, Any

PY = sys.executable  # use same Python interpreter / venv

def _run_cmd(cmd: list[str], timeout: int = 1800) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return {
            "returncode": proc.returncode,
            "cmd": " ".join(cmd),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "cmd": " ".join(cmd), "stdout": "", "stderr": f"timeout after {timeout}s"}
    except Exception as e:
        return {"returncode": -2, "cmd": " ".join(cmd), "stdout": "", "stderr": str(e) + "\n" + traceback.format_exc()}

# --- existing script wrappers ---
def upload_raw_to_mongo(local_data: str = "data/synthetic_dataset", limit: int = 0, mongo_uri: Optional[str] = None, mongo_db: Optional[str] = None, gridfs: bool = False) -> Dict[str, Any]:
    cmd = [PY, os.path.join("src","upload_raw_to_mongo.py"), "--data", local_data, "--out", "output/ingestion", "--gridfs", "False"]
    if limit and int(limit)>0:
        cmd += ["--limit", str(limit)]
    if mongo_uri:
        cmd += ["--mongo-uri", mongo_uri]
    if mongo_db:
        cmd += ["--mongo-db", mongo_db]
    return _run_cmd(cmd, timeout=900)

def create_canonical_profiles(limit: int = 0, applicant: Optional[str] = None, llm_model: str = "mistral", out: str = "output/canonical_profiles", local_data: str = "data/synthetic_dataset", mongo_uri: Optional[str] = None, mongo_db: Optional[str] = None) -> Dict[str, Any]:
    cmd = [PY, os.path.join("src","create_canonical_profiles.py"), "--llm-model", llm_model, "--out", out, "--local-data", local_data]
    if limit and int(limit)>0:
        cmd += ["--limit", str(limit)]
    if applicant:
        cmd += ["--applicant", applicant]
    if mongo_uri:
        cmd += ["--mongo-uri", mongo_uri]
    if mongo_db:
        cmd += ["--mongo-db", mongo_db]
    return _run_cmd(cmd, timeout=900)

# --- new: run pydantic validation runner (calls small script run_validation.py) ---
def run_pydantic_validation(limit: int = 0, applicants_file: Optional[str] = None, mongo_uri: Optional[str] = None, mongo_db: Optional[str] = None) -> Dict[str, Any]:
    """
    Calls src/validation/run_validation.py which imports schema_v2.validate_and_store_v2
    and runs validation over canonical_profiles in Mongo (or a provided list).
    """
    cmd = [PY, os.path.join("src","validation","run_validation.py")]
    if limit and int(limit)>0:
        cmd += ["--limit", str(limit)]
    if applicants_file:
        cmd += ["--applicants-file", applicants_file]
    if mongo_uri:
        cmd += ["--mongo-uri", mongo_uri]
    if mongo_db:
        cmd += ["--mongo-db", mongo_db]
    return _run_cmd(cmd, timeout=900)

def export_training_table(mongo_uri: Optional[str] = None, mongo_db: Optional[str] = None, limit: int = 0, applicants_file: Optional[str] = None, out: str = "output/training", split_train_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Calls src/export_training_table.py to create training CSVs and feature files.
    """
    cmd = [PY, os.path.join("src","export_training_table.py"), "--out", out]
    if limit and int(limit)>0:
        cmd += ["--limit", str(limit)]
    if applicants_file:
        cmd += ["--applicants-file", applicants_file]
    if split_train_size:
        cmd += ["--split-train-size", str(split_train_size)]
    if mongo_uri:
        cmd += ["--mongo-uri", mongo_uri]
    if mongo_db:
        cmd += ["--mongo-db", mongo_db]
    return _run_cmd(cmd, timeout=900)

def train_xgb(train_csv: str = "output/training/train.csv", test_csv: str = "output/training/test.csv", labels_csv: str = "data/synthetic_dataset/labels.csv", out_models: str = "models", eval_out: str = "output/evaluation", compute_shap: bool = False, seed: int = 42) -> Dict[str, Any]:
    cmd = [PY, os.path.join("src","train_xgb.py"), "--train", train_csv, "--test", test_csv, "--labels", labels_csv, "--out", out_models, "--eval-out", eval_out, "--seed", str(seed)]
    if compute_shap:
        cmd += ["--compute-shap"]
    return _run_cmd(cmd, timeout=1800)

def run_mistral_explainer(applicant: Optional[str] = None, llm_model: str = "mistral", out: str = "output/explanations", local_data: str = "data/synthetic_dataset", mongo_uri: Optional[str] = None, mongo_db: Optional[str] = None) -> Dict[str, Any]:
    explainer_path = os.path.join("src","llm_explainer_mistral.py")
    if os.path.exists(explainer_path):
        cmd = [PY, explainer_path, "--llm-model", llm_model, "--out", out, "--local-data", local_data]
        if applicant:
            cmd += ["--applicant", applicant]
        if mongo_uri:
            cmd += ["--mongo-uri", mongo_uri]
        if mongo_db:
            cmd += ["--mongo-db", mongo_db]
        return _run_cmd(cmd, timeout=900)
    # fallback: call canonicalizer which often includes llm normalizer
    return create_canonical_profiles(limit=0, applicant=applicant, llm_model=llm_model, out=out, local_data=local_data, mongo_uri=mongo_uri, mongo_db=mongo_db)