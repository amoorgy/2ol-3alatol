"""Level 1 — can a TOKENIZER tell a real spelling variant from a look-alike?

Two arms, always reported side by side (hard rule 1):

  human      data/benchmark/pairs.csv — 102 hand/mined pairs from the
             human-authored seed corpus. The arm any headline must hold on.
  nilechat   data/stimulus/nilechat/stimuli_v1.csv — machine-derived probe
             stimuli: 500 word types, star edges (base->variant SAME,
             base->trap DIFFERENT).

The headline is separability: if fragmentation on DIFFERENT pairs is no higher
than on SAME pairs (AUC ~ 0.5), the tokenizer gives the model no signal for
this distinction and the work must happen at inference time.

Reproducibility is enforced, not hoped for: gated repos are replaced by ungated
mirrors whose token signatures must reproduce the sha pinned from the original,
and a non-optional tokenizer that fails to load aborts the run (exit 2) instead
of silently writing a complete-looking file. `--offline` proves the run works
from cache alone.

    .venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level1_tokenizer.py
"""

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import random
import statistics
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.tokenizer_metrics import (aggregate_by_class, auc,  # noqa: E402
                                           fertility, fragment_edge)

PAIRS = ROOT / "data" / "benchmark" / "pairs.csv"
STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"
# Results live next to the experiment, derived from this script's own
# location so they move with it. Top-level results/ is D1's showcase.
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

BOOT_SEED = 13


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    repo: str
    group: str                    # "production" | "control"
    family: str
    ollama: str | None = None     # the E2 Ollama tag this tokenizer belongs to
    mirror_of: str | None = None  # gated repo this ungated repo stands in for
    expect_signature: str | None = None  # pinned sha the mirror must reproduce
    optional: bool = False        # may skip without failing the run
    note: str = ""


# Signatures are pinned from the committed 2026-08-20 run, which loaded the
# gated originals from a warm local cache. A mirror that reproduces the pin IS
# the original tokenizer for every measurement this script makes.
REGISTRY: tuple[TokenizerSpec, ...] = (
    TokenizerSpec("aya-expanse-8b", "Orion-zhen/aya-expanse-8b-AWQ",
                  "production", "cohere", ollama="aya-expanse:8b",
                  mirror_of="CohereLabs/aya-expanse-8b",
                  expect_signature="34c655bfe58a"),
    TokenizerSpec("aya-expanse-32b", "mlx-community/aya-expanse-32b-8bit",
                  "production", "cohere",
                  mirror_of="CohereLabs/aya-expanse-32b",
                  expect_signature="34c655bfe58a"),
    TokenizerSpec("command-r7b-arabic", "CohereLabs/c4ai-command-r7b-arabic-02-2025",
                  "production", "cohere", optional=True,
                  expect_signature="34c655bfe58a",
                  note="gated, no ungated mirror exists; redundant — the "
                       "committed run proved it shares aya-expanse's tokenizer"),
    TokenizerSpec("command-a", "CohereLabs/c4ai-command-a-03-2025",
                  "production", "cohere", optional=True,
                  expect_signature="34c655bfe58a",
                  note="gated, no ungated mirror; same signature as aya-expanse"),
    TokenizerSpec("gemma-4-e2b", "google/gemma-4-e2b-it",
                  "production", "google", ollama="gemma4:e2b"),
    TokenizerSpec("qwen3.5-9b", "Qwen/Qwen3.5-9B",
                  "production", "qwen", ollama="qwen3.5:9b"),
    TokenizerSpec("llama-3.1-8b", "unsloth/Meta-Llama-3.1-8B",
                  "production", "meta", ollama="llama3.1:8b",
                  note="stands in for gated meta-llama/Llama-3.1-8B, which "
                       "never loaded in any committed run, so no signature "
                       "pin exists; unsloth republishes the checkpoint verbatim"),
    TokenizerSpec("nilechat-3b", "ranwakhaled/nilechat-3b-ideal-noquant",
                  "production", "dialect",
                  mirror_of="UBC-NLP/NileChat-3B",
                  expect_signature="ba9132633c65",
                  note="the only model pretrained on the exact probe corpus"),
    TokenizerSpec("mistral-7b", "mistralai/Mistral-7B-v0.3",
                  "production", "mistral", expect_signature="3293be6c1cf3"),
    TokenizerSpec("gpt-oss-20b", "openai/gpt-oss-20b",
                  "production", "openai", expect_signature="a68642ca9c81"),
    TokenizerSpec("byt5", "google/byt5-small", "control", "byte-level",
                  expect_signature="41b681325a10",
                  note="no vocabulary at all: fragmentation bounded by edit "
                       "distance. The ceiling that scales every other row"),
    TokenizerSpec("mt5-base", "google/mt5-base", "control", "unigram-lm",
                  note="SentencePiece-unigram control: different segmentation "
                       "ALGORITHM, not just different data. Replaces aya-101, "
                       "which carried this same mT5 vocabulary behind a gate"),
)


# --- edges -------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeEdge:
    a: str
    b: str
    label: str          # SAME | DIFFERENT
    subcategory: str
    classes: tuple[str, ...]


def human_edges(path: pathlib.Path) -> list[ProbeEdge]:
    with path.open(encoding="utf-8") as handle:
        return [
            ProbeEdge(row["a"], row["b"], row["label"], row["subcategory"],
                      tuple(c for c in row["ortho_classes"].split("+")
                            if c and c != "none"))
            for row in csv.DictReader(handle)
        ]


def nilechat_edges(path: pathlib.Path) -> list[ProbeEdge]:
    """Star, not clique: 4 SAME edges + up to 1 DIFFERENT edge per item.

    A clique would emit 10 SAME edges of uncontrolled compound difficulty per
    item and let single items dominate the AUC.
    """
    by_item: dict[str, dict[str, list[dict]]] = collections.defaultdict(
        lambda: {"base": [], "variant": [], "trap": []})
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_item[row["item_id"]][row["role"]].append(row)

    edges: list[ProbeEdge] = []
    for item in sorted(by_item):
        roles = by_item[item]
        if not roles["base"]:
            continue
        base = roles["base"][0]["rendering"]
        for row in roles["variant"]:
            if row["council_valid"] == "false":
                continue  # council-invalidated: kept in the file, not scored
            edges.append(ProbeEdge(
                base, row["rendering"], "SAME", row["licence"],
                tuple(c for c in row["ortho_classes"].split("+") if c)))
        for row in roles["trap"]:
            if row["council_valid"] == "false":
                continue
            edges.append(ProbeEdge(base, row["rendering"], "DIFFERENT",
                                   "trap", ()))
    return edges


# --- measurement -------------------------------------------------------------

def signature(tok, words: list[str]) -> str:
    ids = [list(tok.encode(w, add_special_tokens=False)) for w in words]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]


def bootstrap_auc(diff: list[float], same: list[float],
                  n_boot: int) -> tuple[float, float] | None:
    if not diff or not same:
        return None
    rng = random.Random(BOOT_SEED)
    scores = sorted(
        auc([rng.choice(diff) for _ in diff], [rng.choice(same) for _ in same])
        for _ in range(n_boot)
    )
    return (round(scores[int(0.025 * n_boot)], 4),
            round(scores[int(0.975 * n_boot)], 4))


def score_arm(encode, edges: list[ProbeEdge], n_boot: int) -> dict:
    words = sorted({w for e in edges for w in (e.a, e.b)})
    scored = [(e, fragment_edge(encode, e.a, e.b, classes=e.classes))
              for e in edges]
    same = [f.frag for e, f in scored if e.label == "SAME"]
    diff = [f.frag for e, f in scored if e.label == "DIFFERENT"]

    by_subcategory: dict[str, list[float]] = collections.defaultdict(list)
    for e, f in scored:
        by_subcategory[f"{e.label.lower()}/{e.subcategory}"].append(f.frag)

    sep = auc(diff, same) if diff and same else None
    return {
        "n_edges": len(edges),
        "n_same": len(same),
        "n_different": len(diff),
        "fertility": round(fertility(encode, words), 3),
        "mean_frag_same": round(statistics.fmean(same), 4) if same else None,
        "mean_frag_different": (round(statistics.fmean(diff), 4)
                                if diff else None),
        "separability_auc": round(sep, 4) if sep is not None else None,
        "separability_ci95": bootstrap_auc(diff, same, n_boot),
        # Per-phenomenon fragmentation over SAME edges only: DIFFERENT edges
        # carry a subcategory (lexical/trap/...), not orthographic classes.
        "by_class": [dataclasses.asdict(s) for s in aggregate_by_class(
            f for e, f in scored if e.label == "SAME")],
        "by_subcategory": {k: {"mean_frag": round(statistics.fmean(v), 4),
                               "n": len(v)}
                           for k, v in sorted(by_subcategory.items())},
    }


# --- output ------------------------------------------------------------------

def markdown(payload: dict) -> str:
    lines = [
        "# Level 1 — can a tokenizer tell a real variant from a look-alike?",
        "",
        "**Provenance.** The `human` arm is human-authored "
        "(`data/benchmark/pairs.csv`); the `nilechat` arm is machine-derived "
        "probe stimulus (Command R+ transliteration) and is never a benchmark "
        "artifact. Headlines must hold on the human arm.",
        "",
        "`separability AUC` = probability a DIFFERENT pair fragments more than "
        "a SAME pair; 0.5 = the tokenizer offers the model no signal.",
        "",
    ]
    for arm, arm_meta in payload["arms"].items():
        lines += [
            f"## Arm: {arm} ({arm_meta['provenance']}, "
            f"{arm_meta['n_same']} SAME / {arm_meta['n_different']} DIFFERENT)",
            "",
            "| tokenizer | vocab | frag SAME | frag DIFFERENT | AUC | 95% CI |",
            "|---|---|---|---|---|---|",
        ]
        rows = [(name, entry) for name, entry in payload["tokenizers"].items()
                if arm in entry["arms"]]
        rows.sort(key=lambda kv: kv[1]["arms"][arm]["separability_auc"] or 0.5)
        for name, entry in rows:
            r = entry["arms"][arm]
            ci = r["separability_ci95"]
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(
                f"| {name} | {entry['vocab_size']} | {fmt(r['mean_frag_same'])} "
                f"| {fmt(r['mean_frag_different'])} | **{fmt(r['separability_auc'])}** "
                f"| {f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else 'n/a'} |")
        lines.append("")
    lines += ["## Models sharing a tokenizer", ""]
    for sig, members in payload["tokenizer_identity"].items():
        if len(members) > 1:
            lines.append(f"- `{sig}`: {', '.join(members)}")
    lines += ["", "## Mirrors standing in for gated repos", "",
              "| tokenizer | repo used | stands in for | pinned signature | verified |",
              "|---|---|---|---|---|"]
    for name, entry in payload["tokenizers"].items():
        if entry.get("mirror_of"):
            lines.append(f"| {name} | {entry['repo']} | {entry['mirror_of']} "
                         f"| `{entry['signature']}` "
                         f"| {'yes' if entry['signature_verified'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("human", "nilechat", "both"),
                        default="both")
    parser.add_argument("--pairs", type=pathlib.Path, default=PAIRS)
    parser.add_argument("--stimuli", type=pathlib.Path, default=STIMULI)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--only", default="",
                        help="comma-separated registry subset (smoke: --only byt5)")
    parser.add_argument("--allow-skip", action="store_true",
                        help="tolerate load failures on non-optional rows. Off "
                             "by default: a silent skip is how an unreproducible "
                             "results file gets written without anyone noticing")
    parser.add_argument("--offline", action="store_true",
                        help="HF_HUB_OFFLINE=1: prove the run works from cache")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    registry = [s for s in REGISTRY if not wanted or s.name in wanted]

    arms: dict[str, list[ProbeEdge]] = {}
    if args.arm in ("human", "both"):
        arms["human"] = human_edges(args.pairs)
    if args.arm in ("nilechat", "both") and args.stimuli.exists():
        arms["nilechat"] = nilechat_edges(args.stimuli)
    elif args.arm in ("nilechat", "both"):
        print(f"note: {args.stimuli} missing; nilechat arm skipped")

    # Signatures are computed over the human-arm word list regardless of which
    # arms run: that is the list the pins were minted on, and it is tracked.
    sig_words = sorted({w for e in human_edges(args.pairs) for w in (e.a, e.b)})

    loaded: dict[str, tuple[TokenizerSpec, object, str]] = {}
    registry_report = []
    for spec in registry:
        status, skip_reason = "loaded", None
        try:
            tok = AutoTokenizer.from_pretrained(spec.repo)
        except Exception as exc:
            text = str(exc).lower()
            skip_reason = ("gated: terms not accepted"
                           if "gated" in text or "403" in text
                           else type(exc).__name__)
            status = "skipped"
            if not spec.optional and not args.allow_skip:
                print(f"FATAL: {spec.name} ({spec.repo}) failed to load: "
                      f"{skip_reason}\nA non-optional tokenizer that cannot "
                      f"load makes the results unreproducible. Fix the repo or "
                      f"pass --allow-skip to accept a partial run explicitly.")
                return 2
            print(f"  skip {spec.name:<20} {skip_reason}")
            registry_report.append({**dataclasses.asdict(spec),
                                    "status": status, "skip_reason": skip_reason})
            continue

        sig = signature(tok, sig_words)
        if spec.expect_signature and sig != spec.expect_signature:
            print(f"FATAL: {spec.name} ({spec.repo}) produced signature {sig}, "
                  f"expected {spec.expect_signature}. The mirror does NOT "
                  f"reproduce the pinned tokenizer; refusing to score it.")
            return 2
        loaded[spec.name] = (spec, tok, sig)
        registry_report.append({**dataclasses.asdict(spec),
                                "status": status, "skip_reason": None})

    identity: dict[str, list[str]] = collections.defaultdict(list)
    for name, (_spec, _tok, sig) in loaded.items():
        identity[sig].append(name)

    print()
    tokenizers: dict[str, dict] = {}
    for sig, members in identity.items():
        name = members[0]
        spec, tok, _ = loaded[name]
        encode = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
        entry = {
            "repo": spec.repo, "family": spec.family, "group": spec.group,
            "ollama": spec.ollama, "mirror_of": spec.mirror_of,
            "signature": sig,
            "signature_verified": bool(spec.expect_signature),
            "shares_tokenizer_with": [m for m in members if m != name],
            "vocab_size": int(tok.vocab_size),
            "tokenizer_class": tok.__class__.__name__,
            "note": spec.note,
            "arms": {arm: score_arm(encode, edges, args.boot)
                     for arm, edges in arms.items()},
        }
        tokenizers[name] = entry
        for arm, r in entry["arms"].items():
            fmt = lambda v: "n/a " if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {name:<18} {arm:<9} frag SAME={fmt(r['mean_frag_same'])} "
                  f"DIFF={fmt(r['mean_frag_different'])} "
                  f"AUC={fmt(r['separability_auc'])}")

    payload = {
        "schema_version": 2,
        "generated_by": "experiments/d5_cohere_track/e1_variant_probe/run_level1_tokenizer.py",
        "arms": {
            arm: {
                "source": str((args.pairs if arm == "human" else args.stimuli)
                              .relative_to(ROOT)),
                "provenance": ("human_authored" if arm == "human"
                               else "machine_derived"),
                "n_edges": len(edges),
                "n_same": sum(e.label == "SAME" for e in edges),
                "n_different": sum(e.label == "DIFFERENT" for e in edges),
            } for arm, edges in arms.items()
        },
        "registry": registry_report,
        "tokenizer_identity": {sig: sorted(m) for sig, m in identity.items()},
        "tokenizers": tokenizers,
    }

    if wanted:
        # A registry subset is a smoke test, never the committed table: write
        # it under a suffixed name so --only cannot clobber the real results.
        suffix = "_subset"
    else:
        suffix = ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"level1_tokenizer{suffix}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / f"level1_tokenizer{suffix}.md").write_text(
        markdown(payload), encoding="utf-8")
    print(f"\nwrote {(args.out_dir / ('level1_tokenizer' + suffix + '.md')).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
