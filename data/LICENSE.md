# Data license

Everything under `data/` is licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The code in the rest of this repository is MIT-licensed (see [`../LICENSE`](../LICENSE)).

Non-commercial is not optional here. Most of this data derives from a
CC BY-NC 4.0 source, and that restriction carries over to anything built from it.

## What derives from what

| path | source | notes |
|---|---|---|
| `seed/` | human-authored Franco messages | the human arm's only text source |
| `benchmark/` | mined from `seed/` plus author-written clusters and annotations | the human-authored test set |
| `stimulus/` (all but the file below) | **NileChat Arabizi-EGY** (CC BY-NC 4.0): probe stimuli, anchor and scenario banks, LLM council verdicts | machine-derived; probe stimuli only, never a benchmark artifact |
| `stimulus/arabic_control_v1.tsv` | author-written | capability-floor control |
| `training/` | **NileChat Arabizi-EGY** (CC BY-NC 4.0), respelled by rule; plus synthetic sentences from the teacher/fusor models listed in [`training/v2/DATASHEET.md`](training/v2/DATASHEET.md) | machine-derived |

Synthetic sentences and council verdicts are model outputs. Anyone reusing them is
also responsible for the terms of the models that produced them; the datasheet
names every model.

## Attribution

NileChat Arabizi-EGY, by UBC-NLP: Egyptian Arabic web text transliterated to
Arabizi with Cohere's Command R+
([dataset](https://huggingface.co/datasets/UBC-NLP/nilechat-arabizi-egy)). Cite:

```bibtex
@inproceedings{el-mekki-etal-2025-nilechat,
    title = "{N}ile{C}hat: Towards Linguistically Diverse and Culturally Aware {LLM}s for Local Communities",
    author = "El Mekki, Abdellah and Atou, Houdaifa and Nacar, Omer and Shehata, Shady and Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.556/",
    doi = "10.18653/v1/2025.emnlp-main.556",
    pages = "10978--11002"
}
```
