from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from adversarial_lassonet.config import apply_config_defaults, load_yaml_config
from scripts.run_sers_lassonet_comparison import build_parser, main


if __name__ == "__main__":
    parser = build_parser()
    config = load_yaml_config("exp1_main_benchmark/sers.yaml")
    apply_config_defaults(parser, config)
    main(parser.parse_args())
