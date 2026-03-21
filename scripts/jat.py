"""Cross-platform launcher for Job Application Tailor.

Usage examples:
  python scripts/jat.py setup
  python scripts/jat.py run
  python scripts/jat.py test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "venv" / "Scripts" / "python.exe"
    return root / "venv" / "bin" / "python"


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def _ensure_venv(root: Path) -> Path:
    vpy = _venv_python(root)
    if vpy.exists():
        return vpy
    print("Creating virtual environment...")
    rc = _run([sys.executable, "-m", "venv", "venv"], cwd=root)
    if rc != 0:
        raise SystemExit(rc)
    return _venv_python(root)


def cmd_setup(args: argparse.Namespace) -> int:
    _ = args
    root = _project_root()
    vpy = _ensure_venv(root)

    print("Installing Python packages...")
    rc = _run([str(vpy), "-m", "pip", "install", "-r", "requirements.txt"], cwd=root)
    if rc != 0:
        return rc

    print("Installing Playwright Chromium...")
    rc = _run([str(vpy), "-m", "playwright", "install", "chromium"], cwd=root)
    if rc != 0:
        return rc

    print("Setup complete.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = _project_root()
    vpy = _venv_python(root)
    if not vpy.exists():
        print("Virtual environment not found. Run: python scripts/jat.py setup")
        return 1

    env = os.environ.copy()
    env["HF_HOME"] = str((root / "data" / "hf_cache").resolve())

    cmd = [
        str(vpy),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload and env.get("JAT_NO_RELOAD") != "1":
        cmd.extend(["--reload", "--reload-dir", "app"])
    return _run(cmd, cwd=root, env=env)


def cmd_test(args: argparse.Namespace) -> int:
    root = _project_root()
    vpy = _venv_python(root)
    if not vpy.exists():
        print("Virtual environment not found. Run: python scripts/jat.py setup")
        return 1
    cmd = [str(vpy), "-m", "pytest", "tests/", "-q"]
    if args.extra:
        cmd.extend(args.extra)
    return _run(cmd, cwd=root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Job Application Tailor launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Create venv and install dependencies")
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="Run the FastAPI app with uvicorn")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", default=8000, type=int)
    p_run.add_argument("--no-reload", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("extra", nargs=argparse.REMAINDER)
    p_test.set_defaults(func=cmd_test)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
