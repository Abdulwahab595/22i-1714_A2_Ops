# 22i-1714_A2_Ops — VisionGuard

[![VisionGuard CI](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/workflows/ci.yml)

Abdul Wahab · 22i-1714. TensorFlow/Keras classification of real and AI-generated images,
with MLflow/DagsHub tracking, DVC, Flask, PostgreSQL, Docker Compose, and a hosted model API.

The model score is experimental and uncalibrated. It is not proof that an image is real or generated.

## Recorded results

The champion is frozen ImageNet MobileNetV2, registered as version 2 with the `champion` alias in
[DagsHub](https://dagshub.com/Abdulwahab595/22i-1714_A2_Ops). Fifteen real-data candidates were compared
using validation data, including eight tuner trials. A later audit found that the initial fine-tuning
attempts selected the augmentation container. Corrected last-block runs and eight corrected tuner
trials are recorded separately; the locked champion and its test evaluation are preserved.

| Locked finalist | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.5782 | 0.7209 | 0.3827 | 0.5000 | 0.6487 |
| MobileNetV2 champion | 0.7279 | 0.7971 | 0.6790 | 0.7333 | 0.8449 |

AI-generated is the positive class. The champion threshold, 0.5626477, was selected on validation data
with real-called-AI errors weighted twice as heavily. The test set contains 147 images and was scored
once per locked finalist. See [result tables and figures](docs/results) and [sample provenance](docs/samples).

## Setup

See [account setup](docs/setup_accounts.md). Copy `.env.example` to `.env` and fill in your own secrets.
Keep `.env` and `.dvc/config.local` out of Git. Python 3.10 is the reference environment.
On Windows, use a short environment path to avoid TensorFlow's deeply nested wheel paths.

```powershell
python -m venv C:\path\to\short\vg310
C:\path\to\short\vg310\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m scripts.environment
python -m scripts.download_dataset
python -m src.audit
```

## Training and evidence

```powershell
python -m scripts.dvc_lab
python -m scripts.toy_mlflow
python -m scripts.toy_tuner
python -m src.train --epochs 10
python -m src.tune --trials 8 --epochs 4
python -m src.evaluate
python -m scripts.register_model
python -m scripts.predict_by_alias path\to\image.jpg
```

`src.train` runs six controlled experiments and uses validation data only. `src.tune` performs
at least eight additional real-data trials. `src.evaluate` locks the two finalists using validation
ROC-AUC before opening test labels, scores each finalist once, and refuses to repeat a completed
evaluation. The decision threshold minimizes `(2 × real-called-AI + AI-called-real) / N` on validation.

If DagsHub setup is delayed, runs are recorded in the local MLflow SQLite store. Publish those genuine
runs with `python -m scripts.sync_mlflow` after configuring DagsHub. The importer preserves metric
timestamps and explicitly identifies the local execution; it does not claim the model was trained remotely.

## Run the application

After setting the pinned `CHAMPION_RUN_ID` and tracking credentials in `.env`, download the released
model once with `python -m scripts.fetch_champion`. Alternatively, use the locally exported model.

```powershell
python app.py
# Or run the production WSGI server:
python serve.py
# Or run web + PostgreSQL:
docker compose up --build -d
docker compose ps
```

Open http://127.0.0.1:8000. `/health` checks the loaded model and database. The UI provides upload,
result and history pages; the API provides `POST /api/predict` with an `image` multipart field,
`GET /api/history`, and `GET /api/predictions/<id>`. If `API_KEY` is set, send `Authorization: Bearer ...`.
Uploads are limited to JPEG/PNG, 5 MB, and two million pixels to bound serving memory.

The Compose database is called `db`, has a health check and a named volume, and publishes no host port.
The application starts after PostgreSQL is ready. The model is mounted read-only into the web container.
Passwords are read from `.env`. Normal shutdown is `docker compose down`; do not add `--volumes` when
you want to preserve prediction history.

For Tasks 7–8, set the local Flask `APP_MODE=remote` and `MODEL_API_URL` to the hosted API's base URL.
The UI then calls that API for prediction and history without loading a local model. The cloud
container uses external PostgreSQL and fetches a pinned champion artifact once at startup.

## Checks

```powershell
python -m ruff check app.py serve.py src scripts tests
python -m pytest -q
```

The test double is restricted to tests; report evaluation and deployed inference use actual trained models.
GitHub Actions installs dependencies, lints, tests and builds the Docker image. CI does not require
access to the private tracking token, training data, or the trained model just to build the image.

## Data provenance

Source: [Cash Bowman's AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images),
dataset version 1, Community Data License Agreement – Sharing – Version 1.0.
The audit records original relative paths, dimensions, format, mode, size, and SHA-256 for every file.
Exact duplicate hashes are removed before stratified splitting; conflicting-label duplicate groups
would be excluded. This does not establish that all near duplicates or source biases have been removed.

The local audit found 975 readable files (539 AI-generated, 436 real), no corrupt files,
and one redundant exact copy. The 974 unique usable images were split into 681 training,
146 validation, and 147 test images. Raw images and model binaries are not committed to Git.
