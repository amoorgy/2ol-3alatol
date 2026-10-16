"""Tiny JSONL + seed-corpus helpers."""

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_messages(paths: Iterable[str | Path], min_len: int = 8) -> list[str]:
    """Unique message texts from chat-format seed files, insertion-ordered."""
    seen: dict[str, None] = {}
    for path in paths:
        for row in read_jsonl(path):
            for message in row.get("messages", []):
                text = (message.get("content") or "").strip()
                if len(text) >= min_len:
                    seen.setdefault(text)
    return list(seen)
