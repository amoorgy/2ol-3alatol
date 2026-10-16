"""Robustness metrics: the benchmark's vocabulary of success and failure."""


def recall_at_1(predictions: list[int | None], gold: list[int]) -> float:
    """Fraction of predictions matching gold. None counts as a miss."""
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold must have equal length")
    if not gold:
        return 0.0
    return sum(p == g for p, g in zip(predictions, gold)) / len(gold)


def consistency_at_k(groups: list[list[int | None]]) -> float:
    """C@k: fraction of groups whose k predictions are all identical.

    A group is the model's predictions on k renderings of the same content.
    Consistency is orthogonal to accuracy — a model can be consistently
    wrong and still count as consistent — but None is a *non-answer*, not an
    answer: a group of total retrieval failures is not "consistent".
    """
    if not groups:
        return 0.0
    return sum(len(set(g)) == 1 and g[0] is not None for g in groups) / len(groups)


def pairwise_agreement(groups: list[list[int | None]]) -> float:
    """Mean fraction of prediction pairs that agree within each group.

    None never agrees with anything, including another None.
    """
    if not groups:
        return 0.0
    scores = []
    for g in groups:
        pairs = [(a, b) for i, a in enumerate(g) for b in g[i + 1 :]]
        if pairs:
            agree = sum(a == b and a is not None for a, b in pairs)
            scores.append(agree / len(pairs))
    return sum(scores) / len(scores) if scores else 0.0
