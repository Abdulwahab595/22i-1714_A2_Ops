"""Prepare real result tables, representative API inputs, and private Render configuration."""
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

HYPOTHESES = [
    "A real-labelled bear painting has vivid unnatural colours and strong brush texture. "
    "The model may be using painterly style as a shortcut for AI generation.",
    "The small portrait has saturated lighting and smooth-looking skin. "
    "Those visual properties may resemble generated portraits after resizing.",
    "The AI-labelled landscape is a stylized stock-style poster with a watermark. "
    "Familiar stock-image presentation may confuse the model; a watermark does not verify origin.",
    "The photograph contains an artist's palette, paint, and brushes. "
    "Painted textures may dominate the features even though the source label is real.",
    "The AI-labelled older-man portrait has plausible lighting and detailed facial texture. "
    "A realistic portrait composition may hide the generator cues learned by this model.",
    "The AI-labelled composite pairs a simple coloured map with a realistic landscape. "
    "The photographic-looking half may dominate a single whole-image decision.",
    "The real-labelled eye is a brightly coloured painting. "
    "Brush patterns and saturated colours may overlap the model's learned AI-style features.",
    "The real portrait has polished lighting and a softly detailed indoor background. "
    "Resizing to 128 pixels may remove subtle photographic detail and make the composition ambiguous.",
    "The real portrait has a vivid repeated-pattern background and strong colour contrasts. "
    "The model may mistake the artistic setting for generated imagery.",
    "The real-labelled group scene has theatrical poses and many small faces. "
    "The crowded composition and loss of facial detail during resizing may contribute to confusion.",
]


def main():
    load_dotenv()
    root = Path("docs/results")
    root.mkdir(parents=True, exist_ok=True)
    errors = Path("artifacts/evaluation/mobilenet_v2/misclassified.csv")
    frame = pd.read_csv(errors)
    if len(frame) != len(HYPOTHESES):
        raise ValueError("Review the actual error images before assigning case-specific hypotheses")
    frame["hypothesis"] = HYPOTHESES
    frame.to_csv(errors, index=False)
    for name in ["test_comparison.csv", "validation_comparison.csv", "locked_selection.json", "registry.json"]:
        shutil.copy2(Path("artifacts/evaluation") / name, root / name)
    for family in ["cnn", "mobilenet_v2"]:
        target = root / family
        target.mkdir(exist_ok=True)
        for name in ["metrics.json", "confusion_matrix.png", "roc_curve.png", "error_grid.png",
                     "misclassified.csv"]:
            shutil.copy2(Path("artifacts/evaluation") / family / name, target / name)
    tuner_root = "artifacts/tuner_corrected" if Path("artifacts/tuner_corrected/trial_results.csv").exists() else "artifacts/tuner"
    for source, target in [(tuner_root + "/trial_results.csv", "tuner_trials.csv"),
                           (tuner_root + "/best_hyperparameters.json", "best_hyperparameters.json"),
                           ("artifacts/toy-tuner/results.csv", "toy_tuner_results.csv")]:
        shutil.copy2(source, root / target)
    samples = Path("docs/samples")
    samples.mkdir(exist_ok=True)
    test = pd.read_csv("artifacts/evaluation/mobilenet_v2/test_predictions.csv")
    provenance = []
    for label, filename in [(0, "real.jpg"), (1, "ai_generated.jpg")]:
        candidates = test[(test["label"] == label) & (test["prediction"] == label)
                          & (test["format"] == "JPEG") & (test["file_size"] < 500_000)
                          & (test["width"] * test["height"] <= 2_000_000)]
        chosen = candidates.sort_values("path").iloc[0]
        shutil.copy2(Path("data/raw") / chosen["path"], samples / filename)
        provenance.append({"file": filename, "original_path": chosen["path"],
                           "label": chosen["class"], "original_sha256": chosen["sha256"],
                           "selection": "Correctly classified demonstration example; not an accuracy estimate"})
    (samples / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (samples / "wrong_type.txt").write_text("A deliberate non-image upload for validation testing.\n")
    (samples / "undecodable.png").write_bytes(b"This is not a decodable PNG image.")
    (samples / "oversized.png").write_bytes(b"x" * (6 * 1024**2))
    (samples / "README.md").write_text(
        "# API demonstration inputs\n\nThe two JPEGs are unchanged public dataset examples. "
        "Their source paths, folder labels, and SHA-256 hashes are recorded in provenance.json. "
        "They were selected as correct examples for the interface demonstration; the reported "
        "accuracy uses all 147 held-out images, including errors.\n\n"
        "Source: https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images\n"
        "License declared by the dataset: Community Data License Agreement – Sharing – Version 1.0.\n"
        "The text, undecodable PNG, and oversized file are intentional validation failures.\n",
        encoding="utf-8")
    metadata = json.loads(Path("models/champion.json").read_text())
    values = {"PROJECT_NAME": "22i-1714_A2_Ops", "APP_MODE": "local", "HOST": "0.0.0.0",
              "PORT": "10000", "MODEL_PATH": "models/champion.keras",
              "MODEL_METADATA_PATH": "models/champion.json", "MAX_UPLOAD_MB": "5",
              "MAX_IMAGE_PIXELS": "2000000", "DATABASE_URL": os.environ["DATABASE_URL"],
              "API_KEY": os.environ.get("API_KEY", ""),
              "MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
              "MLFLOW_TRACKING_USERNAME": os.environ["MLFLOW_TRACKING_USERNAME"],
              "MLFLOW_TRACKING_PASSWORD": os.environ["MLFLOW_TRACKING_PASSWORD"],
              "CHAMPION_RUN_ID": metadata["run_id"]}
    Path(".env.render").write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    current = Path(".env").read_text().splitlines()
    Path(".env").write_text("\n".join(
        "CHAMPION_RUN_ID=" + metadata["run_id"] if line.startswith("CHAMPION_RUN_ID=") else line
        for line in current) + "\n")
    print("Prepared real result tables, ten reviewed error hypotheses, and two traceable sample images.")
    print("Prepared private .env.render; secret values are not printed and the file is Git-ignored.")


if __name__ == "__main__":
    main()
