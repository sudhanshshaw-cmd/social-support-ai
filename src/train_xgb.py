#!/usr/bin/env python3
"""
train_xgb.py

Trains an XGBoost classifier pipeline from CSV features and label file.
- Supports reading a prepared train.csv and test.csv, or a single training table + label file + test-size split.
- Saves: models/pipeline_xgb_v1.joblib, output/evaluation/metrics.json, output/evaluation/feature_importances.json
- Optionally computes SHAP summary (requires shap package).

Usage examples:
  # If you already created train.csv and test.csv via export script:
  python src/train_xgb.py --train output/training/train.csv --test output/training/test.csv --labels data/labels.csv --out models

  # Or read a single table and split:
  python src/train_xgb.py --features output/training/training_table.csv --labels data/labels.csv --test-size 0.2 --out models

Requirements:
  pip install pandas scikit-learn xgboost joblib shap
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb

# optionally import shap if available
try:
    import shap
except Exception:
    shap = None

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "income_source",
    "risk_band",
    "gender",
]

def load_feature_cols_json(p: Path):
    """Return list of feature columns if feature_cols.json exists."""
    try:
        if p.exists():
            return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        pass
    return None

def read_labels(labels_path):
    """
    labels.csv expected columns: applicant_id,label
    label should be 0/1 or 'approve'/'decline' etc.
    """
    df = pd.read_csv(labels_path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    if "applicant_id" not in df.columns:
        raise ValueError("labels csv must contain applicant_id column")
    # detect label column name
    label_col = None
    for c in df.columns:
        if c.lower() in ("label", "label_approve", "y", "target"):
            label_col = c
            break
    if label_col is None:
        if len(df.columns) >= 2:
            label_col = df.columns[1]
        else:
            raise ValueError("labels csv doesn't contain a label column")
    df = df[["applicant_id", label_col]].rename(columns={label_col: "label"})
    # coerce label to 0/1
    def parse_label(v):
        if pd.isna(v): return np.nan
        if str(v).strip() == "": return np.nan
        s = str(v).strip().lower()
        if s in ("1","yes","true","approve","approved","1.0"):
            return 1
        if s in ("0","no","false","decline","rejected","0.0"):
            return 0
        try:
            return int(float(s))
        except:
            return np.nan
    df["label"] = df["label"].apply(parse_label)
    return df.dropna(subset=["label"])

def build_pipeline(numeric_features, categorical_features, xgb_params=None, random_state=42):
    # numeric transformer: impute median + standard scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # categorical transformer: impute constant + onehot
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ], remainder="drop", sparse_threshold=0)

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": random_state
        }

    clf = xgb.XGBClassifier(**xgb_params)
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])
    return pipeline

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test))>1 else None,
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }
    return metrics, probs, preds

def compute_shap(model, X_train, X_test, feature_names, out_dir):
    if shap is None:
        print("[WARN] shap not installed; skipping shap computation")
        return None
    try:
        preproc = model.named_steps["preproc"]
        clf = model.named_steps["clf"]
        X_test_proc = preproc.transform(X_test)
        explainer = shap.Explainer(clf)
        shap_values = explainer(X_test_proc)
        try:
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0).tolist()
        except Exception:
            mean_abs_shap = np.abs(shap_values).mean(axis=0).tolist()
        shap_summary = dict(zip(feature_names[:len(mean_abs_shap)], mean_abs_shap))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "shap_summary.json", "w", encoding="utf-8") as fh:
            json.dump(shap_summary, fh, indent=2)
        return shap_summary
    except Exception as e:
        print(f"[WARN] shap generation failed: {e}")
        return None

def main(args):
    out_dir = Path(args.out) if args.out else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_out) if args.eval_out else Path("output/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Detect feature_cols.json
    features_path = Path(args.features)
    features_dir = features_path.parent if features_path.exists() else Path("output/training")
    feature_cols_json_path = features_dir / "feature_cols.json"
    feature_cols_from_json = load_feature_cols_json(feature_cols_json_path)

    # Read features and labels
    if args.train and args.test:
        train_df = pd.read_csv(args.train)
        test_df = pd.read_csv(args.test)
        labels_df = read_labels(args.labels)
        train_df = train_df.merge(labels_df, on="applicant_id", how="left")
        test_df = test_df.merge(labels_df, on="applicant_id", how="left")
    else:
        feat_df = pd.read_csv(args.features)
        labels_df = read_labels(args.labels)
        df = feat_df.merge(labels_df, on="applicant_id", how="inner")
        if df.empty:
            raise ValueError("No rows after merging features and labels. Check applicant IDs.")
        if args.test_size and args.test_size > 0:
            train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, shuffle=False)
        else:
            if args.train_size and args.train_size > 0:
                train_df = df.iloc[:args.train_size]
                test_df = df.iloc[args.train_size:]
            else:
                raise ValueError("Provide either --train/--test files or --test-size or --train-size")

    # Prepare features
    drop_cols = ["applicant_id", "created_at", "updated_at", "dob"]
    if feature_cols_from_json:
        feature_cols_ordered = [c for c in feature_cols_from_json if c not in drop_cols and c in train_df.columns]
    else:
        feature_cols = [c for c in train_df.columns if c not in drop_cols and c != "label"]
        feature_cols_ordered = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in feature_cols]
        for c in feature_cols:
            if c not in feature_cols_ordered:
                feature_cols_ordered.append(c)

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Label column missing in train/test. Check labels CSV merge.")

    # --- SAFER: drop rows without labels and coerce to int ---
    # Count and drop unlabeled rows in train
    missing_train = train_df["label"].isna().sum()
    if missing_train > 0:
        print(f"[WARN] {missing_train} rows in train.csv have no label and will be dropped before training.")
    train_df = train_df.dropna(subset=["label"]).copy()

    missing_test = test_df["label"].isna().sum()
    if missing_test > 0:
        print(f"[WARN] {missing_test} rows in test.csv have no label and will be dropped before evaluation.")
    test_df = test_df.dropna(subset=["label"]).copy()

    # now coerce to int (safe)
    y_train = train_df["label"].astype(float).astype(int)
    y_test = test_df["label"].astype(float).astype(int)

    X_train = train_df[feature_cols_ordered].copy()
    X_test = test_df[feature_cols_ordered].copy()

    print(f"[INFO] Training rows: {len(X_train)}  Test rows: {len(X_test)}  Features: {len(feature_cols_ordered)}")

    numeric_present = [c for c in NUMERIC_FEATURES if c in feature_cols_ordered]
    categorical_present = [c for c in CATEGORICAL_FEATURES if c in feature_cols_ordered]
    pipeline = build_pipeline(numeric_features=numeric_present, categorical_features=categorical_present, random_state=args.seed)

    pipeline.fit(X_train, y_train)

    metrics, probs, preds = evaluate_model(pipeline, X_test, y_test)

    joblib.dump(pipeline, out_dir / "pipeline_xgb_v1.joblib")
    print(f"[OK] pipeline saved to {out_dir / 'pipeline_xgb_v1.joblib'}")

    metrics_out = {
        "metrics": metrics,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols_ordered,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2)
    with open(eval_dir / "feature_cols.json", "w", encoding="utf-8") as fh:
        json.dump(feature_cols_ordered, fh, indent=2)

    print(f"[OK] metrics written to {eval_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))

    if args.compute_shap:
        shap_summary = compute_shap(pipeline, X_train, X_test, feature_cols_ordered, eval_dir)
        if shap_summary:
            print(f"[OK] shap summary written to {eval_dir / 'shap_summary.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, default="output/training/training_table.csv", help="Single features CSV (merged) ")
    p.add_argument("--labels", type=str, default="data/labels.csv", help="Labels CSV (applicant_id,label)")
    p.add_argument("--train", type=str, default=None, help="Optional explicit train CSV")
    p.add_argument("--test", type=str, default=None, help="Optional explicit test CSV")
    p.add_argument("--out", type=str, default="models", help="Output folder for pipeline")
    p.add_argument("--eval-out", type=str, default="output/evaluation", help="Output folder for metrics")
    p.add_argument("--test-size", type=float, default=0.2, help="If using single features file, test split size")
    p.add_argument("--train-size", type=int, default=0, help="Alternative: use first N rows as train")
    p.add_argument("--compute-shap", action="store_true", help="Try to compute shap summary (requires shap)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)