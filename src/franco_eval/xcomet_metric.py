"""XCOMET-XL meaning-fidelity scoring for Dataset B candidates.

Reference-free mode: source = the anchor's own Arabic script + gloss
(e.g. "قائمة (a list)"), hypothesis = the generated Franco-Arabizi
candidate sentence directly -- verified live that omitting "ref" entirely
gives the same score as passing an empty one (0.578 vs 0.580 on a test
pair), so no back-gloss step or forced reference is needed. This is a
stretch of COMET's intended use (built for genuine cross-lingual MT
evaluation, not monolingual generation around a fixed word) but close
enough to "does this rendering convey the source meaning" to be
informative.

Not Ollama: this is a local HF/PyTorch regression model loaded via the
`unbabel-comet` package, in its OWN venv (.venv-comet, Python 3.12) --
unbabel-comet pins huggingface-hub<1.0, incompatible with the main .venv's
huggingface_hub>=1.28, and its pinned torchmetrics==0.10.3 imports private
functools internals removed in Python 3.14. Run scripts using this module
with `.venv-comet/bin/python`, not `.venv/bin/python`.

num_workers is pinned to 1, not left at comet's None/0 default: comet
unconditionally sets multiprocessing_context="fork" whenever Apple's MPS
backend is available (comet/models/base.py), which crashes with
num_workers=0 on any Apple Silicon Mac -- a bug in comet's code, not ours.

Loading the checkpoint takes real time and several GB of RAM; do not run
this concurrently with heavy Ollama generation (verified live: doing so
exhausted this machine's 32GB RAM into swap and silently killed both
processes).

    .venv-comet/bin/python -c "
    from franco_eval.xcomet_metric import score_batch
    print(score_batch([('قائمة (a list)', 'Ana 3ayez 2a2ema el-mowaa3ed')]))"
"""

import functools

CHECKPOINT = "Unbabel/XCOMET-XL"


@functools.lru_cache(maxsize=1)
def _model():
    from comet import download_model, load_from_checkpoint
    path = download_model(CHECKPOINT)
    return load_from_checkpoint(path)


def score_batch(pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
    """(source, hypothesis) pairs -> COMET quality scores, roughly in [0, 1]."""
    data = [{"src": source, "mt": hypothesis} for source, hypothesis in pairs]
    output = _model().predict(data, batch_size=batch_size, gpus=0,
                              num_workers=1, progress_bar=False)
    return [float(s) for s in output.scores]
