from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Fill this in with the path to loss_history.csv
input_path = ""

if not input_path:
    raise ValueError("Please set input_path to your loss_history.csv file.")

csv_path = Path(input_path)
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)
print("Columns:", df.columns.tolist())

required_cols = ["epoch", "train_loss", "val_loss"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

plt.figure()
plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss vs epoch")
plt.legend()
plt.tight_layout()

out_path = csv_path.with_name("train_val_loss_vs_epoch.png")
plt.savefig(out_path, dpi=200)

print("Saved:", out_path)
