"""Read results JSON into plain structures. STDLIB ONLY."""

import json
import pathlib


def load_json(path: pathlib.Path) -> dict | None:
    """None when the file is absent -- the renderer skips, it never invents."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
