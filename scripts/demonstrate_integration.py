"""Start the local UI and demonstrate genuine local-Flask-to-cloud predictions."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    base = "http://127.0.0.1:8001"
    cloud = os.environ["MODEL_API_URL"].rstrip("/")
    if not cloud.startswith("https://"):
        raise ValueError("MODEL_API_URL must be the hosted HTTPS API address")
    target = Path("artifacts/local_cloud_integration.json")
    target.parent.mkdir(exist_ok=True)
    report = {"project": "22i-1714_A2_Ops", "time_utc": datetime.now(timezone.utc).isoformat(),
              "local_ui_url": base, "hosted_api_url": cloud, "requests": []}
    session = requests.Session()
    key = os.getenv("API_KEY", "")
    if key:
        session.headers["Authorization"] = "Bearer " + key
    try:
        try:
            running = session.get(base + "/health", timeout=2)
            ready = running.status_code == 200 and running.json().get("mode") == "remote"
        except requests.RequestException:
            ready = False
        if not ready:
            env = os.environ.copy()
            env.update(APP_MODE="remote", HOST="127.0.0.1", PORT="8001", PYTHONIOENCODING="utf-8")
            with Path("artifacts/local_cloud_server.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-u", "serve.py"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            report["local_server_pid"] = child.pid
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError("Local server exited; inspect artifacts/local_cloud_server.log")
                try:
                    health = session.get(base + "/health", timeout=5)
                    if health.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Local UI did not become ready; inspect the server log")
        health = session.get(base + "/health", timeout=120)
        health.raise_for_status()
        report["health"] = health.json()
        assert report["health"]["mode"] == "remote"
        assert report["health"]["upstream"] == "healthy"
        print("22i-1714_A2_Ops: local Flask is connected to the hosted model API.", flush=True)
        ids = []
        for name in ["real.jpg", "ai_generated.jpg"]:
            with (Path("docs/samples") / name).open("rb") as image:
                response = session.post(base + "/", files={"image": image}, timeout=120)
            response.raise_for_status()
            if "/result/" not in response.url:
                raise AssertionError("The upload did not produce a local result page")
            identifier = response.url.rsplit("/", 1)[-1]
            record = session.get(base + "/api/predictions/" + identifier, timeout=60)
            record.raise_for_status()
            body = record.json()
            assert body["id"] == identifier and body["status"] == "success"
            ids.append(identifier)
            report["requests"].append({"file": name, "result_page": response.url,
                                        "http_status": response.status_code, "response": body})
            print(f"{name}: {body['label']}, score {body['confidence']:.4f}, "
                  f"model {body['model_version']}", flush=True)
            print("Result page:", response.url, flush=True)
        history = session.get(base + "/api/history", timeout=60)
        history.raise_for_status()
        assert set(ids) <= {row["id"] for row in history.json()["predictions"]}
        history_page = session.get(base + "/history", timeout=60)
        history_page.raise_for_status()
        assert all(identifier in history_page.text for identifier in ids)
        report.update(status="passed", history_verified=True)
        print("Cloud database history verified for both local uploads.", flush=True)
        print("Open the local UI:", base, flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
