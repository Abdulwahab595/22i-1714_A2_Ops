# Evidence provenance

All reported scores come from actual trained models and saved runs. No evaluation was rerun
while assembling the report. Captions distinguish the following capture types.

- `environment_terminal.png`, `render_*.png`, `flask_health.png`, `flask_upload_cloud.png`,
  `external_api_terminal.png`, and `dagshub_project.png`: screenshots supplied by the student.
- `github_*.jpg`, `docker_actions_green.jpg`, `external_api_green.jpg`: real public GitHub run pages.
- `mlflow_local_*.jpg`: actual local MLflow toy runs and comparison.
- `dagshub_*.jpg` except the project screenshot: installed MLflow 2.22 UI displaying the real
  DagsHub REST API through a local read-only proxy. These runs and registry entries were not recreated.
- `flask_*_local.jpg`, `flask_remote_*.jpg`: actual local Flask pages. The remote examples used
  the hosted model API and its persistent PostgreSQL history.
- `flask_error_*.jpg`: replay of exact saved public-service HTML responses, with stylesheet URLs
  resolved to the original public service. Actual HTTP status and time are recorded in
  `artifacts/ui_validation_responses.json`.
- `dvc_*.jpg`, `cnn_summary_terminal.jpg`, `alias_terminal.jpg`, `docker_warmup.jpg`,
  `docker_compose_healthy.jpg`, `docker_failure_recovery.jpg`, `docker_image_sizes.jpg`, and
  `postman_results.jpg`: clearly labeled archival views of genuine saved execution output.
  They are not screenshots of newly executed terminal sessions.

The Docker source is `artifacts/labs/docker/commands.txt` from successful workflow 42476319549.
Image sizes and layer records are alongside it. DVC source is `artifacts/labs/dvc_lab.txt`.
Alias predictions are preserved in `artifacts/labs/alias_predictions.json`. The model summary and
training figures are attached to MLflow runs and included as small files in the source archive.
Newman results are sanitized in `artifacts/postman_results.json`; raw output and private environments
are excluded because they contain the API credential. Screenshots do not expose secret values.
