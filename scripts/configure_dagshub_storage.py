"""Configure public DVC endpoint and private local credentials without exposing a token."""
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from dvc.config import Config


if __name__ == "__main__":
    load_dotenv()
    owner = os.environ["DAGSHUB_USERNAME"]
    repository = os.environ.get("DAGSHUB_REPO", "22i-1714_A2_Ops")
    token = os.getenv("DAGSHUB_TOKEN") or os.environ["MLFLOW_TRACKING_PASSWORD"]
    if not token:
        raise ValueError("Set a DagsHub token in .env first")
    os.environ.setdefault("DVC_SITE_CACHE_DIR", str(Path("../../work/dvc-site-cache").resolve()))
    os.environ.setdefault("DVC_NO_ANALYTICS", "1")
    config = Config(dvc_dir=".dvc")
    with config.edit("repo") as public:
        public.setdefault("remote", {})["dagshub"] = {
            "url": "s3://dvc", "endpointurl": f"https://dagshub.com/{owner}/{repository}.s3"}
        public.setdefault("core", {})["remote"] = "dagshub"
    with config.edit("local") as private:
        private.setdefault("remote", {})["dagshub"] = {
            "access_key_id": token, "secret_access_key": token}
    print("DagsHub remote configured; credentials remain in ignored .dvc/config.local.", flush=True)
    subprocess.run([sys.executable, "-m", "dvc", "remote", "list"], check=True)
    subprocess.run([sys.executable, "-m", "dvc", "push", "-r", "dagshub", "-j", "4"], check=True)
