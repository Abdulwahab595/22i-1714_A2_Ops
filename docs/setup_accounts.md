# Accounts and setup for 22i-1714_A2_Ops

Student: Abdul Wahab, roll number 22i-1714.

1. Sign in to GitHub and create an empty public repository named `22i-1714_A2_Ops`.
   Leave the README, license, and gitignore options empty because this project already supplies files.
2. Sign in to DagsHub using GitHub. Connect the GitHub repository and retain the same repository name.
   Create a DagsHub access token in account settings. Enter it locally in `.env` as
   `MLFLOW_TRACKING_PASSWORD` and `DAGSHUB_TOKEN`; do not paste it into chat or commit it.
   Set the username and `MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/22i-1714_A2_Ops.mlflow`.
3. Open Docker Desktop and wait until the Linux engine is running. No Docker Hub account is required
   to build and run this project locally; Docker Hub authentication may be requested for pull limits.
4. Sign in to Render using GitHub and choose the free web-service plan when deploying this repository.
   Do not select a paid plan by accident. The supplied `render.yaml` describes the deployment.
5. Sign in to Neon using GitHub and create a free PostgreSQL project. Use `22i-1714_A2_Ops` as its
   display name. Copy the connection string into the Render `DATABASE_URL` secret, including
   `sslmode=require`. The application accepts `postgresql://` and uses pg8000 internally with verified TLS.
6. Configure the Render environment variables listed in `render.yaml`. The champion run ID is
   produced by model registration. The server downloads that exact model once before startup,
   validates its SHA-256, then loads it locally. It does not contact DagsHub during predictions.
7. Use the Render service URL as `MODEL_API_URL` in the local `.env` and change `APP_MODE=remote`.
   Match `MODEL_API_TOKEN` to the cloud API key if one is configured. This runs the Flask interface
   locally while prediction and database history live in the cloud.

The exact assignment project name is retained in app metadata, experiments, registry, GitHub, DagsHub,
and the report. Docker image/project names and DNS hostnames use lowercase `22i-1714_a2_ops` or
hyphenated `22i-1714-a2-ops` because those platforms restrict uppercase characters.

The public Kaggle dataset download is automated; a Kaggle account is only needed if Kaggle changes
its public-download access. A university email alone is not a cloud account.

Free-plan limits must be checked in the provider dashboard before deployment. Render can sleep after
idle time and has limited memory; Neon storage and compute are metered within its free allowance.
Sources: https://render.com/docs/free and https://neon.com/pricing (checked 2026-11-08).
