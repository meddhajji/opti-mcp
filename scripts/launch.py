"""Start the optimizer MCP server, installing its dependencies on first run.

Claude Code starts this with the system Python. It keeps a private virtual
environment in ${CLAUDE_PLUGIN_DATA} (kept across plugin updates) holding
only the dependencies from requirements.txt, reinstalled when that file
changes. The plugin's own code runs straight from the plugin folder, so
nothing is ever written there. Everything pip prints goes to stderr:
stdout is the MCP channel and must carry protocol messages only.
"""
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CLAUDE_PLUGIN_DATA") or ROOT / ".plugin-data")
ENV = DATA / "venv"
STAMP = ENV / "optimizer-installed.txt"


def env_python() -> Path:
    return ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


REQUIREMENTS = ROOT / "requirements.txt"


def code_fingerprint() -> str:
    """Changes whenever the dependency list changes."""
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def ensure_installed() -> Path:
    fingerprint = code_fingerprint()
    py = env_python()
    if py.exists() and STAMP.exists() and STAMP.read_text() == fingerprint:
        return py
    if sys.version_info < (3, 10):
        sys.exit("optimizer plugin: Python 3.10 or newer is required")
    if not py.exists():
        print("optimizer plugin: creating its Python environment (first run only)...", file=sys.stderr)
        venv.create(ENV, with_pip=True)
    print("optimizer plugin: installing OR-Tools and the MCP SDK (about a minute the first time)...", file=sys.stderr)
    subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
        check=True, stdout=sys.stderr,
    )
    STAMP.write_text(fingerprint)
    return py


def main():
    py = ensure_installed()
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    # stdin/stdout pass straight through to the server process
    sys.exit(subprocess.run([str(py), "-m", "optimizer.server"], env=env).returncode)


if __name__ == "__main__":
    main()
