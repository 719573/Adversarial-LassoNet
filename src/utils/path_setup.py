from pathlib import Path
import sys


def add_legacy_src_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "src"
    extra_paths = [
        src_dir / "models",
        src_dir / "utils",
        src_dir / "models" / "lassonet_sers",
    ]
    for path in extra_paths:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
