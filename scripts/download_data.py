"""Download the public Kaggle archive without requiring a Kaggle account token."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path


URL = "https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter"


def main() -> None:
    destination = Path("data/raw/kaggle")
    target = destination / "twcs" / "twcs.csv"
    if target.exists():
        print(f"Already present: {target}")
        return
    print("Downloading Kaggle archive (about 177 MB)...")
    with urllib.request.urlopen(URL, timeout=180) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        zipped.extractall(destination)
    if not target.exists():
        raise RuntimeError(f"Expected {target} after download, but it was not found.")
    print(f"Extracted {target}")


if __name__ == "__main__":
    main()
