"""Genuine warm-up, image comparison, Compose operations, and failure diagnosis evidence."""
import argparse
import getpass
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()
    output = Path("artifacts/labs/docker")
    output.mkdir(parents=True, exist_ok=True)
    transcript = (output / "commands.txt").open("w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        transcript.write(message + "\n")
        transcript.flush()

    def run(*command, expected=0):
        log(f"\n{getpass.getuser()}@22i-1714_A2_Ops {Path.cwd()}\n> " + " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        log(result.stdout + result.stderr)
        if expected == "failure":
            assert result.returncode != 0, "The deliberate failure unexpectedly succeeded"
            return result.stdout + result.stderr
        elif result.returncode != expected:
            raise RuntimeError(f"Command exited {result.returncode}: {' '.join(command)}")
        return result.stdout

    log(f"22i-1714_A2_Ops | Docker lab | {datetime.now(timezone.utc).isoformat()}")
    run("docker", "version")
    suffix = str(int(time.time()))
    name, volume = f"visionguard-warmup-{suffix}", f"visionguard-warmup-data-{suffix}"
    run("docker", "pull", "nginx:alpine")
    run("docker", "volume", "create", volume)
    run("docker", "run", "-d", "--name", name, "-p", "127.0.0.1:18080:80",
        "-v", f"{volume}:/usr/share/nginx/html", "nginx:alpine")
    run("docker", "exec", name, "sh", "-c",
        "echo '22i-1714_A2_Ops Docker warm-up' > /usr/share/nginx/html/index.html")
    response = requests.get("http://127.0.0.1:18080", timeout=15)
    response.raise_for_status()
    log(f"Mapped-port HTTP response: {response.status_code} {response.text.strip()}")
    run("docker", "logs", name)
    run("docker", "exec", name, "sh", "-c", "id; ls /usr/share/nginx/html")
    run("docker", "ps", "-a")
    run("docker", "images")
    run("docker", "stop", name)
    run("docker", "rm", name)
    alias = f"22i-1714_a2_ops:warmup-{suffix}"
    run("docker", "tag", "nginx:alpine", alias)
    run("docker", "image", "rm", alias)
    run("docker", "volume", "rm", volume)
    if args.warmup_only:
        transcript.close()
        return
    run("docker", "build", "-f", "Dockerfile.unoptimized", "-t", "22i-1714_a2_ops:before", ".")
    run("docker", "build", "-t", "22i-1714_a2_ops:web", ".")
    sizes = {}
    for tag in ["before", "web"]:
        size = int(run("docker", "image", "inspect", f"22i-1714_a2_ops:{tag}",
                       "--format", "{{.Size}}"))
        sizes[tag] = {"bytes": size, "MiB": size / 1024**2}
        history = run("docker", "history", "--no-trunc", f"22i-1714_a2_ops:{tag}")
        (output / f"layers_{tag}.txt").write_text(history)
    (output / "image_sizes.json").write_text(json.dumps(sizes, indent=2))
    run("docker", "compose", "up", "-d", "--no-build")
    for _ in range(90):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        run("docker", "compose", "logs", "--no-color")
        raise RuntimeError("Compose never became healthy")
    log("Healthy Compose stack: " + response.text)
    headers = {"Authorization": "Bearer " + os.getenv("API_KEY", "")}
    sample = Path("docs/samples/real.jpg")
    with sample.open("rb") as image:
        response = requests.post("http://127.0.0.1:8000/api/predict", files={"image": image},
                                 headers=headers, timeout=60)
    response.raise_for_status()
    prediction_id = response.json()["id"]
    log("Prediction before service restart: " + response.text)
    run("docker", "compose", "ps")
    for service in ["web", "db"]:
        run("docker", "compose", "exec", "-T", service, "sh", "-c", "id; pwd")
        run("docker", "compose", "logs", "--no-color", "--tail", "30", service)
    database_id = run("docker", "compose", "ps", "-q", "db").strip()
    bindings = json.loads(run("docker", "inspect", database_id,
                              "--format", "{{json .HostConfig.PortBindings}}"))
    assert not bindings, "The database unexpectedly publishes a host port"
    (output / "database_port_bindings.json").write_text(json.dumps(bindings))
    run("docker", "compose", "stop")
    run("docker", "compose", "start")
    run("docker", "compose", "restart", "web")
    for _ in range(60):
        try:
            response = requests.get("http://127.0.0.1:8000/api/history", headers=headers, timeout=5)
            if response.status_code == 200:
                assert prediction_id in {row["id"] for row in response.json()["predictions"]}
                log("Database persistence verified after stopping and restarting both services.")
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Could not verify history after restart")
    # A one-off container points at an absent model. The healthy model and history are preserved.
    failure = run("docker", "compose", "run", "--rm", "--no-deps", "-e",
                  "MODEL_PATH=/app/models/intentionally_missing.keras", "web", expected="failure")
    (output / "deliberate_failure_stdout.txt").write_text(failure)
    log("Diagnosis from logs: the one-off container was configured with a missing model file. "
        "Fix: use /app/models/champion.keras, the read-only model mount in compose.yaml.")
    run("docker", "compose", "up", "-d", "--no-build")
    run("docker", "compose", "logs", "--no-color", "--tail", "25", "web")
    transcript.close()


if __name__ == "__main__":
    main()
