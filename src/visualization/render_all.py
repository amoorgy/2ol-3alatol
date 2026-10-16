"""Render every figure whose inputs exist.

    .venv/bin/python -m visualization.render_all \
        --results-dir experiments/d5_cohere_track/results

Figures with missing inputs are skipped with a reason (exit 0) unless
--fail-on-missing. Output: <figure_id>.<mode>.<ext> plus figures/index.md
naming each figure, the claim it supports, and its inputs.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from visualization.loaders import load_json


# figure id -> (claim, needs, builder(name->payload) -> Figure|None)
def _catalog():
    from visualization import figures_e1, figures_e2, figures_joint

    return {
        "f1_fragmentation_ceiling": (
            "Every production tokenizer shreds Arabizi variants; the byte-level "
            "control shows how much of that is avoidable.",
            ("level1",),
            lambda data, mode: figures_e1.fig_fragmentation_ceiling(
                data["level1"], arm="human", mode=mode)),
        "f2_separability_auc": (
            "Fragmentation barely separates real variants from look-alikes; "
            "the CI band shows what is indistinguishable from chance.",
            ("level1",),
            lambda data, mode: figures_e1.fig_separability_auc(
                data["level1"], arm="human", mode=mode)),
        "f3_phenomenon_heatmap": (
            "Pre-registered: DIGIT_SAFE fragments worst among single-edit "
            "classes, FINAL_Y_I least.",
            ("level1",),
            lambda data, mode: figures_e1.fig_phenomenon_heatmap(
                data["level1"], arm="nilechat", mode=mode)),
        "f4_arm_dumbbell": (
            "Hard rule 1 discharged visually: the headline holds (or not) on "
            "the human-authored arm.",
            ("level1",),
            lambda data, mode: figures_e1.fig_arm_dumbbell(
                data["level1"], mode=mode)),
        "f5_robustness_scatter": (
            "Under-robustness and over-normalisation are separate failures; "
            "no model is good at both.",
            ("level3",),
            lambda data, mode: figures_e2.fig_robustness_scatter(
                data["level3"], mode=mode)),
        "f6_agreement_ladder": (
            "C@5 floors out on free-form Arabic; pairwise agreement is the "
            "informative statistic at k=5.",
            ("level3",),
            lambda data, mode: figures_e2.fig_agreement_ladder(
                data["level3"], mode=mode)),
        "f7_flip_by_phenomenon": (
            "Spelling-induced comprehension failure concentrates in specific "
            "phenomena.",
            ("level3",),
            lambda data, mode: figures_e2.fig_flip_by_phenomenon(
                data["level3"], mode=mode)),
        "f8_capability_floor": (
            "These numbers are about orthography, not a general "
            "inability to explain Egyptian words in Arabic.",
            ("level3",),
            lambda data, mode: figures_e2.fig_capability_floor(
                data["level3"], mode=mode)),
        "f9_frag_vs_flip": (
            "The pre-registered E4 question, drawn whatever the answer is.",
            ("level1", "level3"),
            lambda data, mode: figures_joint.fig_frag_vs_flip(
                data["level1"], data["level3"], mode=mode)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--mode", default="light,dark")
    parser.add_argument("--format", default="svg")
    parser.add_argument("--only", default="")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "level1": load_json(args.results_dir / "level1_tokenizer.json"),
        "level3": load_json(args.results_dir / "level3_meaning.json"),
    }
    only = {n.strip() for n in args.only.split(",") if n.strip()}
    modes = [m.strip() for m in args.mode.split(",")]
    formats = [f.strip() for f in args.format.split(",")]

    index = ["# Figures", ""]
    missing = 0
    for figure_id, (claim, needs, build) in _catalog().items():
        if only and figure_id not in only:
            continue
        absent = [n for n in needs if data.get(n) is None]
        if absent:
            print(f"skip {figure_id}: missing {', '.join(absent)}")
            missing += 1
            continue
        for mode in modes:
            figure = build(data, mode)
            if figure is None:
                print(f"skip {figure_id} ({mode}): no data inside inputs")
                missing += 1
                break
            for fmt in formats:
                path = out_dir / f"{figure_id}.{mode}.{fmt}"
                figure.savefig(path, dpi=args.dpi, metadata=(
                    {"Date": None} if fmt == "svg" else None))
                print(f"wrote {path}")
            import matplotlib.pyplot as plt
            plt.close(figure)
        index += [f"## {figure_id}", "", f"**Claim:** {claim}",
                  f"**Inputs:** {', '.join(needs)}", ""]

    (out_dir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    if missing and args.fail_on_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
