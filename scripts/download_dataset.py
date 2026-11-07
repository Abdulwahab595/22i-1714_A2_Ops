"""Download the public assignment dataset without embedding account credentials."""
import argparse
import json
import time
import zipfile
from pathlib import Path

import requests

REF = "cashbowman/ai-generated-images-vs-real-images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--archive", default="data/dataset.zip")
    args = parser.parse_args()
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(f"https://www.kaggle.com/api/v1/datasets/view/{REF}", timeout=60)
    metadata.raise_for_status()
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/dataset_source.json").write_text(
        json.dumps(metadata.json(), indent=2), encoding="utf-8"
    )
    if not archive.exists() or not zipfile.is_zipfile(archive):
        partial = archive.with_suffix(".zip.part")
        with requests.get(
            f"https://www.kaggle.com/api/v1/datasets/download/{REF}", stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            downloaded, started, last_report = 0, time.monotonic(), 0
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= 50 * 1024 * 1024:
                        print(f"Downloaded {downloaded / 1e6:.0f} MB in "
                              f"{time.monotonic() - started:.0f}s", flush=True)
                        last_report = downloaded
        partial.replace(archive)
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            resolved = (target / entry.filename).resolve()
            if target not in resolved.parents and resolved != target:
                raise ValueError("Archive contains an unsafe path")
        package.extractall(target)
        print(f"Extracted {len(package.infolist())} entries to {target}", flush=True)


if __name__ == "__main__":
    main()
