"""Download standard time series benchmark datasets from Hugging Face."""

from huggingface_hub import hf_hub_download
import os

DEST = "data"
os.makedirs(DEST, exist_ok=True)

# The thuml/Time-Series-Library datasets are mirrored on HF
# by several users. Try the most common mirror.
datasets = {
    "weather.csv": ("thuml/Time-Series-Library", "dataset/weather/weather.csv"),
    "electricity.csv": ("thuml/Time-Series-Library", "dataset/electricity/electricity.csv"),
    "traffic.csv": ("thuml/Time-Series-Library", "dataset/traffic/traffic.csv"),
}

for name, (repo, path) in datasets.items():
    dest_path = os.path.join(DEST, name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        print(f"Skipping {name} (already exists)")
        continue
    print(f"Downloading {name} from {repo}...")
    try:
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=path,
            repo_type="model",
            local_dir=DEST,
            local_dir_use_symlinks=False,
        )
        # Move from nested path to flat
        if os.path.exists(downloaded) and downloaded != dest_path:
            os.rename(downloaded, dest_path)
        print(f"  -> {dest_path} ({os.path.getsize(dest_path)} bytes)")
    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternative: direct dataset repo
        try:
            downloaded = hf_hub_download(
                repo_id="ethanchen/time-series-datasets",
                filename=name,
                repo_type="dataset",
                local_dir=DEST,
                local_dir_use_symlinks=False,
            )
            print(f"  -> {dest_path} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e2:
            print(f"  Also failed: {e2}")
