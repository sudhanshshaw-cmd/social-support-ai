python - <<'PY'
import pandas as pd
from pathlib import Path

src = Path("data/synthetic_dataset/labels.csv")
dst = Path("data/labels.csv")

if not src.exists():
    raise SystemExit(f"Source labels file not found: {src}")

df = pd.read_csv(src, dtype=str).fillna("")
cols = [c.strip() for c in df.columns]
df.columns = cols

# attempt to find the decision column name
dec_col = None
for c in cols:
    if c.lower() in ("decision","decision_result","status"):
        dec_col = c
        break
if dec_col is None:
    # fallback: try second column
    if len(cols) >= 2:
        dec_col = cols[1]
    else:
        raise SystemExit("Couldn't find a decision column in labels CSV")

def map_decision(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s in ("approved","approve","yes","1","accepted","approved "):
        return 1
    if s in ("declined","decline","no","0","rejected","rejected "):
        return 0
    # some files may use other words - try to guess
    if "approve" in s: return 1
    if "declin" in s or "reject" in s: return 0
    return None

df["label"] = df[dec_col].apply(map_decision)

# drop rows without a mapped label
before = len(df)
df = df.dropna(subset=["label"]).copy()
after = len(df)
print(f"[INFO] Read {src} rows={before}; mapped & kept {after} rows with numeric labels.")

# Keep only applicant_id and label (as 0/1 int)
app_col = None
for c in cols:
    if c.lower() in ("applicant_id","app_id","id"):
        app_col = c
        break
if app_col is None:
    # fallback: first column
    app_col = cols[0]

out = df[[app_col, "label"]].rename(columns={app_col: "applicant_id"})
out["label"] = out["label"].astype(int)
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print(f"[OK] Wrote numeric labels to {dst} (rows={len(out)})")
