# Submission guide - Abdul Wahab, 22i-1714

Submit `22i-1714_VisionGuard.pdf` and the requested links. A separate source archive is supplied
for convenience; the GitHub repository remains the versioned source. The front page of the PDF
records the graded source commit and the verified deployed application revision separately.

| Item | Location |
|---|---|
| GitHub | https://github.com/Abdulwahab595/22i-1714_A2_Ops |
| DagsHub | https://dagshub.com/Abdulwahab595/22i-1714_A2_Ops |
| Live browser interface | https://two2i-1714-a2-ops.onrender.com |
| API health | https://two2i-1714-a2-ops.onrender.com/health |
| Prediction API | POST https://two2i-1714-a2-ops.onrender.com/api/predict |

For a browser demonstration, open the live interface, upload either example in `docs/samples/`,
and open History. The public service may need time to wake after inactivity.

For direct API calls, import `docs/postman/VisionGuard.postman_collection.json` into Postman.
Set `base_url` to the service URL and `api_key` privately. Use the supplied example image paths
from the project root. Credentials are not part of the submission archive or report.

To run the archive with Docker, extract it, copy `.env.example` to `.env`, choose your own
`POSTGRES_PASSWORD`, and run `docker compose up --build -d` from the extracted project folder.
The archive includes `models/champion.keras`; the Git repository fetches it through the documented
pinned-model process. Open http://127.0.0.1:8000. Shut down with `docker compose down` to retain history.

For the required local Flask-to-cloud demonstration, configure the untracked `.env` with
`APP_MODE=remote`, `HOST=127.0.0.1`, `PORT=8001`, the hosted `MODEL_API_URL`, and the matching private
`MODEL_API_TOKEN`. Run `python serve.py`, or `python -m scripts.demonstrate_integration` to submit
the two known examples and check their cloud records. Remote mode needs no local model or database.

The report states the actual limitations: small dataset and uncalibrated scores, a late fine-tuning
correction kept separate from the original once-only test evaluation, remote Docker execution,
archived execution-record screenshots, and the inconsistent DagsHub code listing. Direct DagsHub
Git refs, experiment records, model registry and dataset storage were checked independently.
Optional Grad-CAM was omitted. No instructor page limit was supplied in the assignment text.

Do not upload `.env`, `.env.render`, tokens, private runner output, database passwords, or caches.
