"""Audit every file, quarantine ambiguous hashes, split unique images, and produce EDA."""
import argparse
import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def infer_class(path):
    parts = {p.lower().replace("-", "").replace("_", "") for p in path.parts[:-1]}
    if parts & {"aiartdata", "ai", "aigenerated", "fake", "synthetic"}:
        return "ai_generated", 1
    if parts & {"realart", "real", "realimages", "authentic"}:
        return "real", 0
    return "unknown", -1


def inspect_file(path, root):
    relative = path.relative_to(root)
    name, label = infer_class(relative)
    row = {"path": relative.as_posix(), "class": name, "label": label,
           "file_size": path.stat().st_size, "width": None, "height": None,
           "mode": None, "format": None, "sha256": "", "status": "ok", "error": ""}
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        row["sha256"] = digest.hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                row.update(width=image.width, height=image.height, mode=image.mode,
                           format=image.format)
                image.load()
        if label == -1:
            row["status"] = "unknown_class"
    except Exception as error:
        row["status"] = "unreadable"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def create_eda(valid, raw_root, output, seed):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    for row, name in enumerate(["real", "ai_generated"]):
        group = valid[valid["class"] == name]
        samples = group.sample(min(6, len(group)), random_state=seed)
        for ax, (_, entry) in zip(axes[row], samples.iterrows()):
            with Image.open(raw_root / entry["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                ax.imshow(image)
            ax.set_title(f"{name}\n{entry['width']:.0f} x {entry['height']:.0f}", fontsize=9)
            ax.axis("off")
    fig.suptitle("22i-1714_A2_Ops | Representative source images", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "samples.png", dpi=150)
    plt.close(fig)
    captions = []
    for column, label in [("width", "Width (pixels)"), ("height", "Height (pixels)"),
                          ("file_size", "File size (bytes)")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = valid[column].to_numpy()
        bins = np.geomspace(max(1, values.min()), values.max() + 1, 35)
        for name, color in [("real", "#3460d5"), ("ai_generated", "#d55a34")]:
            subset = valid.loc[valid["class"] == name, column]
            ax.hist(subset, bins=bins, alpha=0.6, label=name, color=color)
        ax.set_xscale("log")
        ax.set(xlabel=label + " (log scale)", ylabel="Number of images",
               title=f"22i-1714_A2_Ops | {label} by class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{column}_histogram.png", dpi=150)
        plt.close(fig)
        medians = valid.groupby("class")[column].median().to_dict()
        captions.append({"figure": f"{column}_histogram.png", "class_medians": medians,
                         "interpretation": f"The median {column} is {medians.get('real', 0):.0f} "
                         f"for real and {medians.get('ai_generated', 0):.0f} for AI-generated images. "
                         "A size difference may become a shortcut for the classifier, so both "
                         "classes use the same resizing policy; resize alone does not prove "
                         "that source or compression bias has been removed."})
    (output / "figure_captions.json").write_text(json.dumps(captions, indent=2))
    del rng


def run(raw_root, output, split_dir, seed=42):
    raw_root, output, split_dir = Path(raw_root), Path(output), Path(split_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in raw_root.rglob("*") if p.is_file() and p.name != ".DS_Store")
    if not paths:
        raise ValueError(f"No files found under {raw_root}")
    print(f"Auditing {len(paths)} files", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(lambda p: inspect_file(p, raw_root), paths))
    audit = pd.DataFrame(rows).sort_values("path")
    readable = audit[audit["status"] == "ok"].copy()
    counts = readable.groupby("sha256")["label"].nunique()
    conflicting = set(counts[counts > 1].index)
    audit["duplicate"] = audit.duplicated("sha256", keep=False) & audit["sha256"].ne("")
    audit["conflicting_label"] = audit["sha256"].isin(conflicting)
    audit.to_csv(output / "image_audit.csv", index=False)
    audit[audit["duplicate"]].to_csv(output / "duplicates.csv", index=False)
    valid = readable[~readable["sha256"].isin(conflicting)].drop_duplicates("sha256").copy()
    if valid["label"].nunique() != 2:
        raise ValueError(f"Expected two recognized classes, found {valid['class'].unique()}")
    train, other = train_test_split(valid, test_size=0.3, stratify=valid["label"],
                                   random_state=seed)
    validation, test = train_test_split(other, test_size=0.5, stratify=other["label"],
                                       random_state=seed)
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame.sort_values("path").to_csv(split_dir / f"{name}.csv", index=False)
    stats = readable.groupby("class")[["width", "height", "file_size"]].agg(
        ["min", "median", "max"])
    stats.to_csv(output / "class_statistics.csv")
    summary = {
        "all_files": len(audit), "readable_counts": readable["class"].value_counts().to_dict(),
        "unreadable_files": int(audit["status"].eq("unreadable").sum()),
        "unknown_class_files": int(audit["status"].eq("unknown_class").sum()),
        "duplicate_groups": int(audit.loc[audit["duplicate"], "sha256"].nunique()),
        "duplicate_files_including_first": int(audit["duplicate"].sum()),
        "redundant_readable_copies": int(len(readable) - readable["sha256"].nunique()),
        "conflicting_hash_groups": len(conflicting), "unique_usable_images": len(valid),
        "formats": readable["format"].value_counts().to_dict(),
        "modes": readable["mode"].value_counts().to_dict(), "seed": seed,
        "split_counts": {name: {str(k): int(v) for k, v in frame["class"].value_counts().items()}
                         for name, frame in [("train", train), ("validation", validation),
                                             ("test", test)]},
        "audit_sha256": hashlib.sha256((output / "image_audit.csv").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    create_eda(valid, raw_root, output, seed)
    print(json.dumps(summary, indent=2), flush=True)
    print(stats.to_string(), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="artifacts/audit")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.raw, args.output, args.splits, args.seed)
