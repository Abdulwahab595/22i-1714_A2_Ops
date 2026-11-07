"""Call the real deployment, save responses, and verify that predictions reach history."""
import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/deployment_verification.json")
    args = parser.parse_args()
    base = os.environ["MODEL_API_URL"].rstrip("/")
    session = requests.Session()
    if os.getenv("MODEL_API_TOKEN"):
        session.headers["Authorization"] = "Bearer " + os.environ["MODEL_API_TOKEN"]
    report = {"project": "22i-1714_A2_Ops", "time_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "api_base_url": base,
              "execution_environment": "GitHub-hosted runner" if os.getenv("GITHUB_ACTIONS")
                                        else "local client", "requests": []}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = session.get(base + "/health", timeout=180)
        response.raise_for_status()
        report["health"] = response.json()
        ids = []
        for filename in ["real.jpg", "ai_generated.jpg"]:
            sample = Path("docs/samples") / filename
            with sample.open("rb") as image:
                response = session.post(base + "/api/predict", files={"image": image}, timeout=180)
            response.raise_for_status()
            body = response.json()
            ids.append(body["id"])
            report["requests"].append({"file": filename, "http_status": response.status_code,
                                        "response": body})
        history = session.get(base + "/api/history", timeout=45)
        history.raise_for_status()
        records = history.json()["predictions"]
        recorded_ids = {row["id"] for row in records}
        assert set(ids) <= recorded_ids, "Prediction IDs did not appear in database history"
        report["history_verified"] = True
        report["status"] = "passed"
        print(json.dumps(report, indent=2))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
