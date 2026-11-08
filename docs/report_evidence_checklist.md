# Submission evidence checklist

Only mark an item complete after the actual action and its evidence have been verified.
Final verification: deployment, Docker/Compose, external-network requests, local-to-cloud integration,
Postman/Newman (7 requests, 10 assertions), Ruff and all 19 tests passed on 8 November 2026.
See the JSON records under `artifacts/` and the captioned screenshots under `docs/evidence/`.
The PDF discloses the late backbone correction, original tested release selection, archived-record
screenshots, read-only tracking viewer and DagsHub code-listing inconsistency.

| Task | Evidence to include in the PDF |
|---|---|
| 1.1 | Real terminal versions/GPU output; artifacts/environment.json and packages.txt |
| 1.2 | Full audit CSV, per-class statistics, corrupt/duplicate counts and policy |
| 1.3 | Source sample grid; width/height/file-size histograms with interpretations |
| 1.4 | Toy commits, rollback and restore; real raw.dvc pointer and remote push |
| 2.1 | Split lists, exact RGB/resize/pad explanation and augmented batch |
| 2.2 | Scratch CNN summary, layer shapes and parameter counts |
| 2.3 | Frozen and fine-tuned curves and trainable-layer lists |
| 2.4 | Seeds, callbacks, best epoch on curves |
| 2.5 | Once-only test comparison, confusion matrices, ROC curves, validation threshold policy |
| 2.6 | Ten actual errors, visually inspected image-specific hypotheses |
| 2.7 | Toy tuner results; eight real tuner trials and limits of validation selection |
| 3.1 | Two local MLflow run pages and comparison view; definitions |
| 3.2 | DagsHub repository, remote toy runs, ignored-secret proof |
| 3.3 | Manual and actual successful autologging run pages, params/curves/lineage |
| 3.4 | Artifacts tab showing plots, split CSVs, packages and model |
| 3.5 | At least six meaningful experiments compared using validation scores |
| 3.6 | Registry versions and champion alias; genuine alias-loading prediction output |
| 4.1 | Startup load-once explanation; two known-image inference outputs |
| 4.2 | Wrong type, oversize and undecodable-image friendly errors |
| 4.3 | Upload, result and history frontend screenshots |
| 4.4 | Health response and debug-off server explanation |
| 5 | Persistent database records: time, label, confidence, model version; history view |
| 6.1 | Genuine warm-up commands, mapped port, volume, logs, shell, list/removal |
| 6.2 | Dockerfile, non-root runtime identity, cache-order and CPU-package explanation |
| 6.3 | Healthy Compose stack, database readiness, named volume, no published DB port |
| 6.4 | Actual before/after image sizes and largest layers |
| 6.5 | Start/stop/restart/logs/shell; deliberate failure, log diagnosis and successful fix |
| 6.6 | Separate toy matrix red and green runs; successful project CI and badge |
| 7 | Provider configuration/deployment screenshots, public API URL, Postman results, external-network test |
| 8 | Local Flask calling hosted API, multiple genuine responses and cloud database history |

Each report subsection must explain what was done, why, how it works, and what went wrong.
Every screenshot needs a caption, text reference, and sufficient project/URL/time context.
Never use a test double or generated demonstration score as scientific evaluation evidence.

Current genuine troubleshooting to explain: Windows wheel path length, DVC site-cache permissions,
Windows text encoding for model summaries, and MLflow's immutable batch_size parameter conflict.
Recovered Keras history is explicitly tagged with recovery timestamps; the corrected autologging
run is a separate real training run logged directly to DagsHub.
Additional resolved issues: augmentation-container selection during fine-tuning, a setup-python
runner failure, native psycopg startup crash on Render (replaced with pg8000), and multipart 413
being intercepted by a broad exception handler. Optional Grad-CAM was omitted.
