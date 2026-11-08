"""Publish the two toy DVC commits as a separate branch of the assignment DagsHub project."""
import base64
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    owner = os.environ["DAGSHUB_USERNAME"]
    token = os.getenv("DAGSHUB_TOKEN") or os.environ["MLFLOW_TRACKING_PASSWORD"]
    record = json.loads(Path("artifacts/labs/dvc_lab.json").read_text())
    directory = Path(record["directory"])
    env = os.environ.copy()
    for key in ["GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GCM_TRACE"]:
        env.pop(key, None)
    credential = base64.b64encode(f"{owner}:{token}".encode()).decode()
    env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="http.https://dagshub.com/.extraheader",
               GIT_CONFIG_VALUE_0="Authorization: Basic " + credential,
               GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1="",
               GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="")
    remote = f"https://dagshub.com/{owner}/22i-1714_A2_Ops.git"
    process = subprocess.run(["git", "push", remote, "main:refs/heads/lab-dvc"], cwd=directory,
                              env=env, capture_output=True, text=True, timeout=180)
    print(process.stdout + process.stderr)
    process.check_returncode()
    record["dagshub_repository"] = remote.removesuffix(".git")
    record["dagshub_branch"] = "lab-dvc"
    Path("artifacts/labs/dvc_lab.json").write_text(json.dumps(record, indent=2))
