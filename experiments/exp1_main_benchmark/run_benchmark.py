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
from scripts.run_adversarial_lassonet_benchmark import main, parse_benchmark_args


if __name__ == "__main__":
    parser = parse_benchmark_args(return_parser=True)
    config = load_yaml_config("exp1_main_benchmark/table2.yaml")
    apply_config_defaults(parser, config)
    main(parser.parse_args())
