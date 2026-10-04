"""`factfit dev`: run the FastAPI server and the Next.js UI together, stop both on Ctrl+C.

Written in Python rather than a Makefile so it behaves the same on Windows, macOS and Linux.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def run_dev(api_port: int = 8000, ui_port: int = 3000) -> int:
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found. Install Node.js (https://nodejs.org) and try again.")
        return 1
    if not (UI_DIR / "node_modules").exists():
        print("Installing UI dependencies (first run only)...")
        if subprocess.run([npm, "install"], cwd=UI_DIR).returncode != 0:
            return 1

    api_cmd = [
        sys.executable, "-m", "uvicorn", "factfit.api.app:create_app", "--factory",
        "--reload", "--reload-dir", "factfit", "--port", str(api_port),
    ]  # fmt: skip
    ui_cmd = [npm, "run", "dev", "--", "--port", str(ui_port)]
    ui_env = {**os.environ, "FACTFIT_API_URL": f"http://127.0.0.1:{api_port}"}

    procs = [
        _start(api_cmd, cwd=UI_DIR.parent),
        _start(ui_cmd, cwd=UI_DIR, env=ui_env),
    ]
    print(f"\n  UI:       http://localhost:{ui_port}")
    print(f"  API docs: http://localhost:{api_port}/docs")
    print("  Press Ctrl+C to stop both.\n")

    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        print("One of the servers stopped; stopping the other.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            _stop(p)
    return 0


def _start(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
    if os.name == "nt":
        # New process group so Ctrl+C handling and tree kill stay under our control.
        return subprocess.Popen(
            cmd, cwd=cwd, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    return subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    # npm starts node, and uvicorn --reload starts a worker: kill the whole tree,
    # otherwise a child keeps running and holds the port.
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
