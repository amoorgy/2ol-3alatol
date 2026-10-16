"""franco — a tiny, dependency-free core for Egyptian Arabizi robustness research.

Modules map 1:1 to pipeline steps (and to the specialized agents in
.claude/agents/):

    skeleton   phonemic fingerprint that collapses spelling variants
    taxonomy   attested variation rules with plausibility weights
    perturb    calibrated spelling-variant generator
    normalize  canonical Romanized form ("Franco canon v0")
    mine       attested variant-cluster mining from corpora
    retrieval  char-n-gram TF-IDF retrieval baseline
    metrics    consistency / recall metrics
"""

__version__ = "0.1.0"
