"""Download standard datasets via Nixtla's datasetsforecast."""

import os
from datasetsforecast.long_horizon import LongHorizon

DEST = "data"
os.makedirs(DEST, exist_ok=True)

for name in ["Weather", "ECL", "Traffic"]:
    print(f"Downloading {name}...")
    try:
        Y_df, _, _ = LongHorizon.load(directory=DEST, group=name)
        print(f"  {name}: {len(Y_df)} rows, columns: {list(Y_df.columns)}")
        # Save as CSV for our benchmark harness
        unique_ids = Y_df["unique_id"].unique()
        print(f"  {len(unique_ids)} series")
        # Save the first series as a simple CSV
        first = Y_df[Y_df["unique_id"] == unique_ids[0]].reset_index(drop=True)
        out_path = os.path.join(DEST, f"{name.lower()}_series0.csv")
        first.to_csv(out_path, index=False)
        print(f"  Saved first series to {out_path} ({len(first)} rows)")
    except Exception as e:
        print(f"  Failed: {e}")
