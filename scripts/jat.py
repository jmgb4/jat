"""Cross-platform launcher for Job Application Tailor.

Usage examples:
  python scripts/jat.py setup
  python scripts/jat.py run
  python scripts/jat.py test
  python scripts/jat.py gpu-check
  python scripts/jat.py download-pass-models
  python scripts/jat.py fix-ollama
  python scripts/jat.py commit "optional message suffix"
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
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


def _run_capture(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _ensure_venv(root: Path) -> Path:
    vpy = _venv_python(root)
    if vpy.exists():
        return vpy
    print("Creating virtual environment...")
    rc = _run([sys.executable, "-m", "venv", "venv"], cwd=root)
    if rc != 0:
        raise SystemExit(rc)
    return _venv_python(root)


def _is_nvidia_present(root: Path) -> bool:
    rc, out, _ = _run_capture(["nvidia-smi", "-L"], cwd=root)
    return rc == 0 and bool(out)


def _gpu_requirements_file(root: Path) -> Path | None:
    req_dir = root / "requirements"
    if sys.platform.startswith("win"):
        p = req_dir / "gpu-windows.txt"
    elif sys.platform.startswith("linux"):
        p = req_dir / "gpu-linux.txt"
    elif sys.platform == "darwin":
        p = req_dir / "gpu-macos.txt"
    else:
        return None
    return p if p.exists() else None


def cmd_setup(args: argparse.Namespace) -> int:
    root = _project_root()
    vpy = _ensure_venv(root)

    print("Upgrading pip...")
    rc = _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
    if rc != 0:
        return rc

    base_req = root / "requirements" / "base.txt"
    if not base_req.exists():
        base_req = root / "requirements.txt"

    print(f"Installing base Python packages from {base_req.relative_to(root)}...")
    rc = _run([str(vpy), "-m", "pip", "install", "-r", str(base_req)], cwd=root)
    if rc != 0:
        return rc

    if not args.cpu_only and not args.skip_gpu_stack:
        gpu_req = _gpu_requirements_file(root)
        if gpu_req is not None:
            use_gpu = True
            if sys.platform.startswith(("win", "linux")) and not _is_nvidia_present(root):
                use_gpu = False
                print("NVIDIA GPU not detected. Skipping optional GPU package install.")
            if use_gpu:
                print(f"Installing optional GPU packages from {gpu_req.relative_to(root)}...")
                rc = _run([str(vpy), "-m", "pip", "install", "-r", str(gpu_req)], cwd=root)
                if rc != 0:
                    print("GPU package install failed; continuing with base environment.")
        else:
            print("No platform GPU requirements file found; using base environment only.")
    else:
        print("Skipping GPU package install by request.")

    if not args.skip_playwright:
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
    if args.skip_gpu_check:
        env["JAT_SKIP_GPU_CHECK"] = "1"
        env.setdefault("OLLAMA_NUM_GPU", "0")

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


def cmd_gpu_check(args: argparse.Namespace) -> int:
    _ = args
    root = _project_root()
    vpy = _venv_python(root)
    if not vpy.exists():
        print("Virtual environment not found. Run: python scripts/jat.py setup")
        return 1

    check_script = """
import json
import subprocess
import sys

out = {"platform": sys.platform}

try:
    p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
    out["nvidia_present"] = (p.returncode == 0 and bool((p.stdout or "").strip()))
except Exception:
    out["nvidia_present"] = False

try:
    import torch
    out["torch_cuda"] = bool(torch.cuda.is_available())
except Exception:
    out["torch_cuda"] = False

try:
    from llama_cpp import llama_cpp as lc
    fn = getattr(lc, "llama_supports_gpu_offload", None)
    out["llama_gpu_offload"] = bool(fn() if callable(fn) else False)
except Exception:
    out["llama_gpu_offload"] = False

print(json.dumps(out))
""".strip()
    rc, out, err = _run_capture([str(vpy), "-c", check_script], cwd=root)
    if out:
        try:
            parsed = json.loads(out)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(out)
    if err:
        print(err)
    return rc


def cmd_download_pass_models(args: argparse.Namespace) -> int:
    _ = args
    root = _project_root()
    vpy = _venv_python(root)
    if not vpy.exists():
        print("Virtual environment not found. Run: python scripts/jat.py setup")
        return 1
    script = root / "scripts" / "download_pass_models.py"
    if not script.exists():
        print("Missing script: scripts/download_pass_models.py")
        return 1
    return _run([str(vpy), str(script)], cwd=root)


def _read_ollama_model(root: Path) -> str:
    """Read OLLAMA_MODEL from .env; fall back to qwen2.5:32b."""
    default = "qwen2.5:32b"
    env_file = root / ".env"
    if not env_file.exists():
        return default
    for line in env_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*OLLAMA_MODEL\s*=\s*(.+)", line)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            # Strip "ollama:" prefix if present
            val = re.sub(r"^ollama:", "", val)
            if val:
                return val
    return default


def cmd_fix_ollama(args: argparse.Namespace) -> int:
    """Diagnose Ollama, pull primary model from .env, run a smoke test."""
    _ = args
    root = _project_root()

    if not shutil.which("ollama"):
        print("ERROR: ollama not found on PATH.")
        print("Install from: https://ollama.com/download")
        print("Then restart your shell and re-run this command.")
        return 1

    print("\n=== Ollama Diagnostics ===\n")

    print("1. Installed models (ollama list):")
    _run(["ollama", "list"], cwd=root)

    print("\n2. Running models (ollama ps):")
    _run(["ollama", "ps"], cwd=root)

    model = _read_ollama_model(root)
    print(f"\nPrimary model (from .env): {model}")

    broken = "qwen3-coder:30b"
    print(f"\n3. Removing {broken} if present...")
    rc, _, err = _run_capture(["ollama", "rm", broken], cwd=root)
    if rc != 0:
        print(f"   (not present or already removed: {err})")

    print(f"\n4. Pulling {model} (may take a while on first download)...")
    rc = _run(["ollama", "pull", model], cwd=root)
    if rc != 0:
        print(f"Pull failed for {model}.")
        return rc

    print("\n5. Smoke test...")
    rc, out, err = _run_capture(
        ["ollama", "run", model, "Reply with exactly: READY"], cwd=root
    )
    print(out or err or "(no output)")

    print("\n6. Final model list:")
    _run(["ollama", "list"], cwd=root)

    print(f"\n=== Done ===\nModel '{model}' is ready.")
    print("Start the app with: python scripts/jat.py run")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    """Stage all changes, create a date-stamped commit, and push."""
    root = _project_root()

    if not (root / ".git").exists():
        print("Not a git repository. Run from project root.")
        return 1

    rc, remote, _ = _run_capture(["git", "remote", "get-url", "origin"], cwd=root)
    if rc != 0 or not remote:
        print("No remote 'origin' configured. Add a remote and try again.")
        return 1

    rc, status, _ = _run_capture(["git", "status", "--porcelain"], cwd=root)
    if not status:
        print("No changes to commit. Working tree clean.")
        return 0

    rc = _run(["git", "add", "-A"], cwd=root)
    if rc != 0:
        return rc

    date = datetime.date.today().isoformat()
    suffix = args.message.strip() if args.message else "resume/cover pipeline and style fixes"
    msg = f"Build: {date} - {suffix}"

    rc = _run(["git", "commit", "-m", msg], cwd=root)
    if rc != 0:
        print("Commit failed.")
        return rc

    rc = _run(["git", "push"], cwd=root)
    if rc != 0:
        print("Push failed (network or auth). Fix and run 'git push' manually.")
        return rc

    print(f"Committed and pushed: {msg}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Job Application Tailor launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Create venv and install dependencies")
    p_setup.add_argument("--cpu-only", action="store_true", help="Skip optional GPU package installation")
    p_setup.add_argument("--skip-gpu-stack", action="store_true", help="Skip optional GPU package installation")
    p_setup.add_argument("--skip-playwright", action="store_true", help="Skip playwright browser installation")
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="Run the FastAPI app with uvicorn")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", default=8000, type=int)
    p_run.add_argument("--no-reload", action="store_true")
    p_run.add_argument("--skip-gpu-check", action="store_true", help="Disable startup GPU readiness checks")
    p_run.set_defaults(func=cmd_run)

    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("extra", nargs=argparse.REMAINDER)
    p_test.set_defaults(func=cmd_test)

    p_gpu = sub.add_parser("gpu-check", help="Print local GPU/runtime readiness")
    p_gpu.set_defaults(func=cmd_gpu_check)

    p_models = sub.add_parser("download-pass-models", help="Download and register pass models")
    p_models.set_defaults(func=cmd_download_pass_models)

    p_fix = sub.add_parser("fix-ollama", help="Diagnose Ollama and pull/verify the primary model from .env")
    p_fix.set_defaults(func=cmd_fix_ollama)

    p_commit = sub.add_parser("commit", help="Stage all changes, commit with a date-stamped message, and push")
    p_commit.add_argument("message", nargs="?", default="", help="Optional commit message suffix")
    p_commit.set_defaults(func=cmd_commit)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
