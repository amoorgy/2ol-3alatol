"""Snapshot the reproducibility record for the d5 results directory.

Records the git state, input-file hashes, and package versions that produced
the current results. The input sha256s are what make hard rule 2 *checkable*:
a figure can be traced to the exact stimulus file that fed it.

    .venv/bin/python experiments/d5_cohere_track/write_manifest.py
"""

import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

INPUTS = (
    "data/benchmark/pairs.csv",
    "data/stimulus/nilechat/stimuli_v1.csv",
    "data/stimulus/nilechat/rule_votes.tsv",
    "data/stimulus/nilechat/variant_votes.tsv",
    "data/stimulus/nilechat/trap_votes.tsv",
    "data/stimulus/arabic_control_v1.tsv",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def git(*argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("transformers", "tokenizers", "numpy", "matplotlib",
                 "pyarrow", "sentencepiece"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except ImportError:
            out[name] = "absent"
    return out


def main() -> int:
    manifest = {
        "schema_version": 1,
        "utc": datetime.datetime.now(datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "package_versions": package_versions(),
        "inputs": {p: (sha256(ROOT / p) if (ROOT / p).exists() else "absent")
                   for p in INPUTS},
        "corpus": {
            "name": "UBC-NLP/Arabizi-Egypt (nilechat-arabizi-egy)",
            "provenance": "machine-derived: Command R+ transliteration",
            "licence": "CC-BY-NC-4.0",
        },
        "outputs": sorted(p.name for p in RESULTS.glob("*.json")
                          if p.name != "manifest.json"),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                           encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
