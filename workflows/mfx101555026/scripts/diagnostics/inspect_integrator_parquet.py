from pathlib import Path
import pandas as pd

path = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/"
    "mfx101555026_cctbx/runs/mfx_gamma_qi_25files/"
    "run_20260707-175519_bda0/predictions/test_preds_all.parquet"
)

df = pd.read_parquet(path)

print("FILE:", path)
print("shape:", df.shape)
print("\ncolumns:")
for c in df.columns:
    print(" ", c, df[c].dtype)

print("\nfirst rows:")
print(df.head())

print("\npossible matching columns:")
for c in df.columns:
    if any(k in c.lower() for k in ["h", "k", "l", "miller", "id", "image", "file", "panel", "bbox", "xyz", "d", "intensity", "qi", "epoch"]):
        print(" ", c)

if "epoch" in df.columns:
    print("\nepoch values:")
    print(df["epoch"].value_counts().sort_index())

    final_epoch = df["epoch"].max()
    final_df = df[df["epoch"] == final_epoch]
    print("\nfinal epoch:", final_epoch)
    print("final epoch shape:", final_df.shape)

