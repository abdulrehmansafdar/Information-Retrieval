"""Launch one of the two front ends over the shared search engine.

    python main.py --console              terminal interface
    python main.py --console "query"       one terminal search
    python main.py --web                  Streamlit web interface

Both front ends call engine.py. This file only chooses which interface starts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mini search engine: choose a console or Streamlit interface."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--console", action="store_true",
                       help="run the terminal interface")
    modes.add_argument("--web", action="store_true",
                       help="run the Streamlit web interface")
    args, remaining = parser.parse_known_args(argv)

    if args.console:
        from console_ui import main as console_main

        console_main(remaining)
        return 0

    app_path = ROOT / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", *remaining, str(app_path)]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
