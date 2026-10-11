"""Validate/export the bundled 201-row synthetic student cohort."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
source = ROOT / "data" / "student_dataset_201.csv"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else source
df = pd.read_csv(source)
if len(df) != 201:
    raise ValueError(f"Expected 201 student records, found {len(df)}")
if df["Student ID"].nunique() != 201:
    raise ValueError("Student IDs must be unique")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} student records to {out}")
