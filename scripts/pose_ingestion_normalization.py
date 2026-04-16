from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "video_to_pose.py"
    sys.argv = [target.name, *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
