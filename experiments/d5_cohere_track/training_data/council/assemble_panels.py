"""Assemble one verdict dir from several panels, each owning whole chunks.

A council whose panel changed mid-run -- the v2 pair council ran Opus x-high
for chunks c0-c17, hit the session limit, and finished c18-c22 with Sonnet
medium (user ruling 2026-09-26) -- must still give every item ONE panel's
five lenses. This copies each chunk's lens files from the single panel that
owns it, refuses a chunk claimed twice or a missing lens file, and writes
panel_map.json (chunk -> panel label) so every verdict stays attributable.
The merged dir then goes through merge_council.py --dir unchanged.

    python3 experiments/d5_cohere_track/training_data/council/assemble_panels.py \\
        --out-dir data/generated/v2/council/pair_votes_final \\
        --lenses coherence,identity,plausibility,register,skeptic \\
        --panel opus/xhigh=data/generated/v2/council/pair_votes:c0-c17 \\
        --panel sonnet/medium=data/generated/v2/council/pair_votes_sonnet:c18-c22
"""

import argparse
import json
import pathlib
import re
import shutil

# .meta, not .json: merge_council reads every *.json in a verdict dir as a
# judge file (the stamp is panel.meta for the same reason)
PANEL_MAP = "panel_map.meta"


def parse_chunks(spec: str) -> list[str]:
    """"c0-c2,c5" -> ["c0", "c1", "c2", "c5"]."""
    chunks: list[str] = []
    for part in spec.split(","):
        match = re.fullmatch(r"c(\d+)(?:-c(\d+))?", part.strip())
        if not match:
            raise ValueError(f"bad chunk spec {part!r}")
        lo = int(match.group(1))
        hi = int(match.group(2)) if match.group(2) else lo
        if hi < lo:
            raise ValueError(f"empty chunk range {part!r}")
        chunks.extend(f"c{i}" for i in range(lo, hi + 1))
    return chunks


def assemble(panels: list[tuple[str, pathlib.Path, list[str]]],
             lenses: tuple[str, ...], out_dir: pathlib.Path) -> dict[str, str]:
    owner: dict[str, str] = {}
    for label, _src, chunks in panels:
        for chunk in chunks:
            if chunk in owner:
                raise ValueError(f"{chunk} claimed by both {owner[chunk]} and {label}")
            owner[chunk] = label
    missing = [src / f"{lens}_{chunk}.json" for _label, src, chunks in panels
               for chunk in chunks for lens in lenses
               if not (src / f"{lens}_{chunk}.json").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} lens files missing, e.g. {missing[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for _label, src, chunks in panels:
        for chunk in chunks:
            for lens in lenses:
                shutil.copyfile(src / f"{lens}_{chunk}.json", out_dir / f"{lens}_{chunk}.json")
    ordered = dict(sorted(owner.items(), key=lambda kv: int(kv[0][1:])))
    (out_dir / PANEL_MAP).write_text(json.dumps(ordered, indent=1), encoding="utf-8")
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--lenses", required=True)
    parser.add_argument("--panel", action="append", required=True,
                        help="LABEL=DIR:CHUNKS, e.g. opus/xhigh=dir:c0-c17")
    args = parser.parse_args()
    panels = []
    for spec in args.panel:
        label, _, rest = spec.partition("=")
        directory, _, chunks = rest.rpartition(":")
        panels.append((label, pathlib.Path(directory), parse_chunks(chunks)))
    lenses = tuple(l.strip() for l in args.lenses.split(",") if l.strip())
    owner = assemble(panels, lenses, args.out_dir)
    counts = {label: sum(1 for v in owner.values() if v == label) for label, _d, _c in panels}
    print(f"assembled {len(owner)} chunks x {len(lenses)} lenses into {args.out_dir}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
