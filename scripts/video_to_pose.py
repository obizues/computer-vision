from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_adapters import get_adapter


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical pose records from video or precomputed JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg.get("pose_stage", {}).get("mode", "dataset_json_passthrough")

    output_file = Path(cfg["pose_stage"]["output_top_keypoints_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(mode)
    records = adapter(cfg)

    output_file.write_text(json.dumps(records), encoding="utf-8")
    print(f"Wrote canonical pose records: {output_file} ({len(records)} frames)")


if __name__ == "__main__":
    main()
