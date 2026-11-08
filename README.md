# 22i-1714_A2_Ops — VisionGuard

[![VisionGuard CI](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/workflows/ci.yml)

Abdul Wahab · 22i-1714. TensorFlow/Keras classification of real and AI-generated images,
with MLflow/DagsHub tracking, DVC, Flask, PostgreSQL, Docker Compose, and a hosted model API.

The model score is experimental and uncalibrated. It is not proof that an image is real or generated.

## Submission links and verified operation

- [Live upload interface](https://two2i-1714-a2-ops.onrender.com)
- [Public health check](https://two2i-1714-a2-ops.onrender.com/health)
- Prediction API: `POST https://two2i-1714-a2-ops.onrender.com/api/predict`, multipart field `image`.
- [DagsHub experiments and registry](https://dagshub.com/Abdulwahab595/22i-1714_A2_Ops)

Runtime revision `fe8cec643091260adda25358185fb08118ca285f` passed
[CI](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/runs/42475921813),
[Docker/Compose verification](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/runs/42476319549),
and [a test from a separate GitHub-hosted network](https://github.com/Abdulwahab595/22i-1714_A2_Ops/actions/runs/42476518266).
The Postman collection passed seven requests and ten assertions using Newman. The local Flask UI on
port 8001 submitted two images to the hosted API and confirmed their IDs in cloud history.
Sanitized execution records are in `artifacts/`; captioned evidence is in `docs/evidence/`.
The final local checks passed all 19 tests and Ruff.

Render Free may sleep while idle. Allow a cold start before testing. The public browser UI works
without credentials; direct API testing uses your private key in the Postman environment. Credentials
are deliberately excluded from the repository and submission archive.

## Recorded results

The champion is frozen ImageNet MobileNetV2, registered as version 2 with the `champion` alias in
[DagsHub](https://dagshub.com/Abdulwahab595/22i-1714_A2_Ops). There are 25 real-data attempts:
15 valid configurations and 10 explicitly superseded attempts. A later audit found that the initial fine-tuning
attempts selected the augmentation container. Corrected last-block runs and eight corrected tuner
trials are recorded separately; the locked champion and its test evaluation are preserved.
The corrected candidates improved validation after the original test lock and are validation-only
results. The deployed model is the original tested release, not the highest corrected validation candidate.
See `docs/results/validation_corrected_comparison.csv`; the original comparison remains as historical evidence.

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
PostgreSQL uses the pure-Python pg8000 driver. Neon connections use certificate-verified TLS;
existing PostgreSQL URLs are normalized internally. This avoids the native psycopg connection
crash observed on Render after TensorFlow loaded. Database connection operations have a timeout.

The Compose database is called `db`, has a health check and a named volume, and publishes no host port.
The application starts after PostgreSQL is ready. The model is mounted read-only into the web container.
Passwords are read from `.env`. Normal shutdown is `docker compose down`; do not add `--volumes` when
you want to preserve prediction history.

For Tasks 7–8, set the local Flask `APP_MODE=remote` and `MODEL_API_URL` to the hosted API's base URL.
The UI then calls that API for prediction and history without loading a local model. The cloud
container uses external PostgreSQL and fetches a pinned champion artifact once at startup.
For the demonstrated local-to-cloud setup, use `HOST=127.0.0.1` and `PORT=8001`, then run
`python -m scripts.demonstrate_integration` to submit both supplied examples and check cloud history.

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

## Evidence provenance and known limitations

Some evidence screenshots display archived execution output in a clearly labeled read-only viewer.
Docker commands ran on the linked GitHub Linux runner because the local Docker engine was unavailable.
The tracking screenshots use MLflow 2.22 through a read-only proxy to the real DagsHub API; they show
remote records, not recreated runs. Friendly-error screenshots replay exact saved HTTP responses;
their actual status codes and timestamps are in `artifacts/ui_validation_responses.json`.

DagsHub's code landing page continued to show an empty prompt after synchronization. A direct Git
ref check nevertheless verified `main` at the deployed revision and `lab-dvc` at its actual second
toy commit; see `artifacts/dagshub_git_refs.json`. The source is accessible on GitHub. This web display
limitation is disclosed in the report. DagsHub experiment, model registry and data-storage operations
were verified separately.
