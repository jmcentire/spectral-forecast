"""Upload paper to Zenodo and get DOI."""

import json
import os
import requests

API_KEY = os.environ["ZENODO_API_KEY"]
BASE = "https://zenodo.org/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 1. Create empty deposition
print("Creating deposition...")
r = requests.post(f"{BASE}/deposit/depositions", headers=HEADERS, json={})
r.raise_for_status()
dep = r.json()
dep_id = dep["id"]
bucket_url = dep["links"]["bucket"]
print(f"  Deposition ID: {dep_id}")
print(f"  Bucket: {bucket_url}")

# 2. Upload PDF
print("Uploading PDF...")
with open("paper/paper.pdf", "rb") as f:
    r = requests.put(
        f"{bucket_url}/paper.pdf",
        headers=HEADERS,
        data=f,
    )
    r.raise_for_status()
print(f"  Uploaded: {r.json()['key']}")

# 3. Set metadata
print("Setting metadata...")
metadata = {
    "metadata": {
        "title": "Entropy-Bounded Decomposition for Time Series Forecasting: Fourier Extraction, Adaptive Basis Selection, and the Maximum Entropy Stopping Criterion",
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "We present an analytical time series forecasting method based on iterative "
            "signal decomposition with an information-theoretic stopping criterion. The method "
            "decomposes a time series into periodic components (noise-aware Fourier extraction), "
            "long-period trends (BIC-selected regression), discrete shocks (AIC-selected shape "
            "fitting), and local residual structure (recency-weighted autoregression), then "
            "adaptively applies wavelet decomposition when the Fourier residual exhibits "
            "non-white autocorrelation. On datasets with strong periodic structure (ETTh1, ETTm1), "
            "the method achieves 42-68% improvement over Google's TimesFM (200M parameters). "
            "1,200 lines of Python, no GPU, fully interpretable."
        ),
        "creators": [
            {
                "name": "McEntire, Jeremy",
                "affiliation": "Independent Research",
                "orcid": "",
            }
        ],
        "keywords": [
            "time series forecasting",
            "Fourier decomposition",
            "wavelet analysis",
            "maximum entropy",
            "signal processing",
            "autoregressive models",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/jmcentire/spectral-forecast",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
        "license": "MIT",
    }
}
r = requests.put(
    f"{BASE}/deposit/depositions/{dep_id}",
    headers={**HEADERS, "Content-Type": "application/json"},
    data=json.dumps(metadata),
)
r.raise_for_status()
print("  Metadata set.")

# 4. Publish
print("Publishing...")
r = requests.post(
    f"{BASE}/deposit/depositions/{dep_id}/actions/publish",
    headers=HEADERS,
)
r.raise_for_status()
result = r.json()
doi = result["doi"]
doi_url = result["doi_url"]
print(f"  DOI: {doi}")
print(f"  URL: {doi_url}")
print(f"\nDone! Add this DOI to the repo and perardua.dev.")
