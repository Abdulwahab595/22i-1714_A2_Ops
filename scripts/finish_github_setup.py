"""Run from the user's signed-in terminal to complete the required GitHub labs.

Uses Git Credential Manager without printing or storing its token. Creates one disposable public
CI-lab repository, records a genuine failed and fixed matrix run, encrypts project Actions secrets,
and dispatches the Docker evidence workflow. It does not create paid services or send messages.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import zipfile
import io
from pathlib import Path

import requests
from dotenv import load_dotenv
from nacl import encoding, public

OWNER = "Abdulwahab595"
PROJECT = "22i-1714_A2_Ops"
TOY = PROJECT + "_CI_Lab"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-only", action="store_true")
    parser.add_argument("--secrets-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--verify-url", help="Run external API and final Docker checks, then collect evidence")
    args = parser.parse_args()
    load_dotenv()
    env = os.environ.copy()
    for name in ["GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GCM_TRACE"]:
        env.pop(name, None)
    env.update(GCM_INTERACTIVE="Never", GIT_TERMINAL_PROMPT="0")
    helper = r"C:\Program Files\Git\mingw64\bin\git-credential-manager.exe"
    credential = subprocess.run([helper, "get"], text=True, capture_output=True, env=env,
                                input="protocol=https\nhost=github.com\n\n", timeout=30)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    if credential.returncode or not fields.get("password"):
        raise RuntimeError("GitHub sign-in is unavailable. Run git push in this terminal first.")
    session = requests.Session()
    session.headers.update(Authorization="Bearer " + fields["password"],
                            Accept="application/vnd.github+json",
                            **{"X-GitHub-Api-Version": "2022-11-28"})

    def api(method, path, body=None, allowed=(200, 201, 204)):
        response = session.request(method, "https://api.github.com" + path, json=body, timeout=60)
        if response.status_code not in allowed:
            try:
                message = response.json().get("message", "request failed")
            except ValueError:
                message = "request failed"
            raise RuntimeError(f"GitHub {method} {path}: {response.status_code} {message}")
        return response.json() if response.content else {}

    account = api("GET", "/user")
    if account["login"].lower() != OWNER.lower():
        raise RuntimeError("The signed-in GitHub account does not own this assignment repository")
    auth = base64.b64encode(f"{fields.get('username', OWNER)}:{fields['password']}".encode()).decode()
    env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
               GIT_CONFIG_VALUE_0="Authorization: Basic " + auth,
               GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1="", GIT_ASKPASS="")

    def git(directory, *command):
        process = subprocess.run(["git", *command], cwd=directory, env=env,
                                  capture_output=True, text=True)
        print(process.stdout + process.stderr, flush=True)
        if process.returncode:
            raise RuntimeError("Git operation failed: " + " ".join(command))
        return process.stdout.strip()

    records = Path("artifacts/labs")
    records.mkdir(parents=True, exist_ok=True)

    def wait_for_run(repo, commit, expected):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            runs = api("GET", f"/repos/{OWNER}/{repo}/actions/runs?head_sha={commit}&per_page=5")
            found = runs.get("workflow_runs", [])
            if found and found[0]["status"] == "completed":
                result = found[0]
                if result["conclusion"] != expected:
                    raise RuntimeError(f"Expected {expected}, got {result['conclusion']}: "
                                       + result["html_url"])
                return {k: result[k] for k in ["id", "head_sha", "status", "conclusion", "html_url"]}
            print(f"Waiting for {repo}: expected {expected} ...", flush=True)
            time.sleep(10)
        raise TimeoutError("GitHub Actions did not finish within 15 minutes")

    def collect_run(run):
        if not run:
            return
        identifier = run["id"]
        summary = {key: run.get(key) for key in
                   ["id", "name", "head_sha", "status", "conclusion", "html_url", "created_at"]}
        (records / f"github_run_{identifier}.json").write_text(json.dumps(summary, indent=2))
        listing = api("GET", f"/repos/{OWNER}/{PROJECT}/actions/runs/{identifier}/artifacts")
        for artifact in listing.get("artifacts", []):
            if artifact["expired"]:
                continue
            response = session.get(artifact["archive_download_url"], timeout=90)
            response.raise_for_status()
            destination = Path("artifacts/github") / str(identifier)
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError("Artifact archive contains an unsafe path")
                    if not member.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            for source in destination.rglob("*"):
                if not source.is_file():
                    continue
                if artifact["name"] == "docker-lab-evidence":
                    target = records / "docker" / source.name
                elif source.name == "external-network-results.json":
                    target = Path("artifacts/external-network-results.json")
                else:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        print(f"Collected verified run evidence: {run['html_url']}", flush=True)

    if args.collect_only or args.verify_url:
        workflows = ["docker-evidence.yml", "external-api-test.yml"]
        for workflow in workflows:
            path = f"/repos/{OWNER}/{PROJECT}/actions/workflows/{workflow}/runs?per_page=10"
            before = api("GET", path).get("workflow_runs", [])
            if args.verify_url:
                body = {"ref": "main"}
                if workflow == "external-api-test.yml":
                    body["inputs"] = {"api_url": args.verify_url.rstrip("/")}
                api("POST", f"/repos/{OWNER}/{PROJECT}/actions/workflows/{workflow}/dispatches", body)
                old_ids = {run["id"] for run in before}
                deadline = time.monotonic() + 1500
                while time.monotonic() < deadline:
                    found = api("GET", path).get("workflow_runs", [])
                    new = [run for run in found if run["id"] not in old_ids]
                    if new and new[0]["status"] == "completed":
                        collect_run(new[0])
                        if new[0]["conclusion"] != "success":
                            raise RuntimeError("Verification failed: " + new[0]["html_url"])
                        break
                    print(f"Waiting for {workflow} ...", flush=True)
                    time.sleep(15)
                else:
                    raise TimeoutError("Verification is still running; use --collect-only later")
            else:
                successful = [run for run in before if run["conclusion"] == "success"]
                collect_run(successful[0] if successful else None)
        print("Evidence collected without displaying credentials.", flush=True)
        return

    if not args.secrets_only:
        toy_evidence = records / "toy_ci_runs.json"
        if not toy_evidence.exists():
            existing = session.get(f"https://api.github.com/repos/{OWNER}/{TOY}", timeout=30)
            if existing.status_code == 404:
                api("POST", "/user/repos", {"name": TOY, "private": False, "auto_init": False,
                    "description": "Disposable two-Python CI warm-up for 22i-1714_A2_Ops"})
            elif existing.status_code != 200:
                raise RuntimeError("Could not inspect the disposable CI lab repository")
            directory = (Path.cwd().parent / TOY).resolve()
            if directory.exists() and any(directory.iterdir()) and not (directory / ".git").exists():
                raise RuntimeError("The CI lab directory is not empty; inspect it before continuing")
            directory.mkdir(exist_ok=True)
            if not (directory / ".git").exists():
                git(directory, "init", "-b", "main")
                git(directory, "config", "user.name", "Abdul Wahab")
                git(directory, "config", "user.email", "117485903+Abdulwahab595@users.noreply.github.com")
                git(directory, "remote", "add", "origin", f"https://github.com/{OWNER}/{TOY}.git")
            workflow = directory / ".github/workflows/toy.yml"
            workflow.parent.mkdir(parents=True, exist_ok=True)
            workflow.write_text("""name: Two-Python toy tests
on: [push, workflow_dispatch]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install pytest==8.4.2
      - run: pytest -q
""", encoding="utf-8")
            (directory / "test_toy.py").write_text(
                "from toy import add\n\ndef test_addition():\n    assert add(2, 3) == 5\n")
            (directory / "README.md").write_text(
                "# 22i-1714_A2_Ops CI Lab\n\nAbdul Wahab, 22i-1714. "
                "Two Python versions, an intentional red test, and a corrected green test.\n")
            red_evidence = records / "toy_ci_red.json"
            if red_evidence.exists():
                red = json.loads(red_evidence.read_text())
            else:
                (directory / "toy.py").write_text("def add(a, b):\n    return a - b\n")
                git(directory, "add", ".")
                if git(directory, "status", "--porcelain"):
                    git(directory, "commit", "-m", "Demonstrate failing matrix test")
                red_commit = git(directory, "rev-parse", "HEAD")
                git(directory, "push", "-u", "origin", "main")
                red = wait_for_run(TOY, red_commit, "failure")
                red_evidence.write_text(json.dumps(red, indent=2))
                print("Required RED run recorded:", red["html_url"], flush=True)
            (directory / "toy.py").write_text("def add(a, b):\n    return a + b\n")
            git(directory, "add", "toy.py")
            if git(directory, "status", "--porcelain"):
                git(directory, "commit", "-m", "Fix addition so both Python versions pass")
            green_commit = git(directory, "rev-parse", "HEAD")
            git(directory, "push", "origin", "main")
            green = wait_for_run(TOY, green_commit, "success")
            toy_evidence.write_text(json.dumps({"red": red, "green": green}, indent=2))
            print("Required GREEN run recorded:", green["html_url"], flush=True)
    if args.toy_only:
        return
    git(Path.cwd(), "push", "origin", "main")
    dvc_record_path = records / "dvc_lab.json"
    if dvc_record_path.exists():
        dvc_record = json.loads(dvc_record_path.read_text())
        git(Path(dvc_record["directory"]), "push",
            f"https://github.com/{OWNER}/{PROJECT}.git", "main:refs/heads/lab-dvc")
        dvc_record.update(github_branch="lab-dvc", dagshub_branch="lab-dvc",
                          publication="GitHub branch mirrored by the connected DagsHub repository")
        dvc_record_path.write_text(json.dumps(dvc_record, indent=2))
    registered = json.loads(Path("artifacts/evaluation/registry.json").read_text())
    prefix = f"/repos/{OWNER}/{PROJECT}/actions"
    key = api("GET", prefix + "/secrets/public-key")
    public_key = public.PublicKey(key["key"].encode(), encoding.Base64Encoder())
    box = public.SealedBox(public_key)
    values = {"DAGSHUB_TRACKING_URI": os.environ["MLFLOW_TRACKING_URI"],
              "DAGSHUB_USERNAME": os.environ["MLFLOW_TRACKING_USERNAME"],
              "DAGSHUB_TOKEN": os.environ["MLFLOW_TRACKING_PASSWORD"],
              "MODEL_API_TOKEN": os.getenv("MODEL_API_TOKEN", "")}
    for name, value in values.items():
        encrypted = base64.b64encode(box.encrypt(value.encode())).decode()
        api("PUT", prefix + "/secrets/" + name,
            {"encrypted_value": encrypted, "key_id": key["key_id"]})
        print(f"Configured encrypted Actions secret {name}; value not displayed.", flush=True)
    variable = {"name": "CHAMPION_RUN_ID", "value": registered["champion_run_id"]}
    exists = session.get("https://api.github.com" + prefix + "/variables/CHAMPION_RUN_ID", timeout=30)
    api("PATCH" if exists.status_code == 200 else "POST",
        prefix + ("/variables/CHAMPION_RUN_ID" if exists.status_code == 200 else "/variables"), variable)
    api("POST", prefix + "/workflows/docker-evidence.yml/dispatches", {"ref": "main"})
    print("Dispatched the genuine Docker/Compose evidence run.", flush=True)
    print("GitHub setup complete. Secret values were not printed or written to source files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Setup stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
