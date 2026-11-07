"""Execute the small, isolated DVC exercise and record the real command outputs."""
import argparse
import getpass
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="../../work/labs/dvc-toy")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    os.environ.setdefault("DVC_SITE_CACHE_DIR", str(directory.parent / "dvc-site-cache"))
    os.environ.setdefault("DVC_NO_ANALYTICS", "1")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new empty directory for the DVC toy exercise")
    directory.mkdir(parents=True, exist_ok=True)
    remote = directory.parent / "dvc-toy-remote"
    remote.mkdir(exist_ok=True)
    log_path = Path("artifacts/labs/dvc_lab.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")

    def record(message):
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()

    def run(*command):
        record(f"\n{getpass.getuser()}@22i-1714_A2_Ops {directory}\n> " + " ".join(command))
        process = subprocess.run(command, cwd=directory, capture_output=True, text=True)
        record(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f"Command failed with exit code {process.returncode}")
        return process.stdout.strip()

    record(f"22i-1714_A2_Ops | Abdul Wahab | {datetime.now(timezone.utc).isoformat()}")
    dvc = [sys.executable, "-m", "dvc"]
    run("git", "init", "-b", "main")
    run("git", "config", "user.name", "Abdul Wahab")
    run("git", "config", "user.email", "117485903+Abdulwahab595@users.noreply.github.com")
    run(*dvc, "init")
    run(*dvc, "remote", "add", "-d", "storage", str(remote))
    csv = directory / "images.csv"
    version1 = "image,label\nexample_1.jpg,real\nexample_2.jpg,ai_generated\n"
    csv.write_text(version1)
    run(*dvc, "add", "images.csv")
    run("git", "add", ".")
    run("git", "commit", "-m", "Toy data version 1")
    first = run("git", "rev-parse", "HEAD")
    run(*dvc, "push")
    csv.write_text(version1 + "example_3.jpg,real\n")
    run(*dvc, "add", "images.csv")
    run("git", "add", ".")
    run("git", "commit", "-m", "Toy data version 2")
    second = run("git", "rev-parse", "HEAD")
    run(*dvc, "push")
    run("git", "log", "--oneline")
    run("git", "checkout", first)
    run(*dvc, "checkout")
    if csv.read_text() != version1:
        raise AssertionError("DVC rollback did not recover version 1")
    record("Verified version 1 content after rollback:\n" + csv.read_text())
    run("git", "checkout", "main")
    run(*dvc, "checkout")
    # This is a single known toy file, never a recursive delete or user dataset deletion.
    csv.unlink()
    record("Deleted the toy working CSV only; restoring with dvc pull.")
    run(*dvc, "pull")
    assert "example_3.jpg" in csv.read_text()
    record("Verified version 2 restored:\n" + csv.read_text())
    record("Pointer file:\n" + (directory / "images.csv.dvc").read_text())
    Path("artifacts/labs/dvc_lab.json").write_text(json.dumps({
        "directory": str(directory), "version_1_commit": first, "version_2_commit": second,
        "remote": str(remote), "rollback_verified": True, "restore_verified": True}, indent=2))
    log.close()


if __name__ == "__main__":
    main()
