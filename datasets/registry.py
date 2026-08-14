"""Registry of the datasets used by VEXMLM.

One declarative entry per dataset: identity, source, licence, task, languages,
and how to load it. `datasets/manager.py` consumes this; nothing else should
hardcode dataset paths or Hub IDs.

Availability is honest. `hub_id=None` means no canonical public loader is known,
and the dataset must be supplied via --local-path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    task: str                      # qa | ner | sentiment | mlm
    languages: tuple[str, ...]
    hub_id: str | None
    hub_config: str | None = None
    citation: str = ""
    license: str = "UNKNOWN"
    url: str = ""
    text_field: str | None = None
    label_field: str | None = None
    notes: str = ""
    splits: tuple[str, ...] = ("train", "validation", "test")
    requires_local: bool = False
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- Question Answering ---------------------------------------------------------

TIGQA = DatasetSpec(
    key="tigqa",
    name="TIGQA — Tigrinya Question Answering",
    task="qa",
    languages=("tir",),
    hub_id=None,               # no canonical Hub mirror confirmed
    requires_local=True,
    url="https://arxiv.org/abs/2404.17194",
    citation=(
        "@article{tigqa2024,\n"
        "  title={TIGQA: An Expert-Annotated Question-Answering Dataset "
        "in Tigrinya},\n"
        "  journal={arXiv preprint arXiv:2404.17194},\n"
        "  year={2024}\n}"
    ),
    license="see publication",
    notes=("~2.68K QA pairs over 537 paragraphs. SQuAD-style extractive spans. "
           "Supply with --local-path; no verified Hub mirror."),
)

AMQA = DatasetSpec(
    key="amqa",
    name="AmQA — Amharic Question Answering",
    task="qa",
    languages=("amh",),
    hub_id=None,
    requires_local=True,
    url="https://arxiv.org/abs/2303.03290",
    citation=(
        "@article{amqa2023,\n"
        "  title={An Amharic Question Answering Dataset},\n"
        "  journal={arXiv preprint arXiv:2303.03290},\n"
        "  year={2023}\n}"
    ),
    license="see publication",
    notes=("2,628 QA pairs over 378 Amharic Wikipedia articles. SQuAD-style. "
           "Supply with --local-path."),
)

# --- Named Entity Recognition ---------------------------------------------------

MASAKHANER_AMH = DatasetSpec(
    key="masakhaner_amh",
    name="MasakhaNER 2.0 — Amharic",
    task="ner",
    languages=("amh",),
    hub_id="masakhane/masakhaner2",
    hub_config="amh",
    url="https://github.com/masakhane-io/masakhane-ner",
    citation=(
        "@inproceedings{adelani-etal-2022-masakhaner,\n"
        "  title={MasakhaNER 2.0: Africa-centric Transfer Learning for "
        "Named Entity Recognition},\n"
        "  booktitle={EMNLP},\n  year={2022}\n}"
    ),
    license="CC BY 4.0",
    notes="CoNLL BIO tags: PER, ORG, LOC, DATE.",
)

TIGRINYA_NER = DatasetSpec(
    key="tigrinya_ner",
    name="Tigrinya NER",
    task="ner",
    languages=("tir",),
    hub_id=None,
    requires_local=True,
    url="",
    license="UNKNOWN",
    notes=("Tigrinya is NOT covered by MasakhaNER v1 or v2. The specific corpus "
           "must be supplied with --local-path in CoNLL format. Record its "
           "source in the dataset card before release."),
)

# --- Sentiment ------------------------------------------------------------------

AFRISENTI = DatasetSpec(
    key="afrisenti",
    name="AfriSenti-SemEval 2023 Task 12",
    task="sentiment",
    languages=("amh", "tir", "mul"),
    hub_id="shmuhammad/AfriSenti-twitter-sentiment",
    url="https://github.com/afrisenti-semeval/afrisent-semeval-2023",
    citation=(
        "@inproceedings{muhammad-etal-2023-afrisenti,\n"
        "  title={{A}fri{S}enti: A Twitter Sentiment Analysis Benchmark for "
        "African Languages},\n"
        "  booktitle={SemEval-2023},\n  year={2023}\n}"
    ),
    license="CC BY 4.0",
    text_field="tweet",
    label_field="label",
    notes=("3-class: positive/negative/neutral. Amharic ('amh') is a native "
           "subtask. Tigrinya ('tir') appears in the AfriSenti collection but "
           "not in SemEval Subtask B; verify the config before use."),
)

# --- MLM pretraining corpora ----------------------------------------------------

AMHARIC_CORPUS = DatasetSpec(
    key="amharic_mlm",
    name="Amharic monolingual corpus (Stage 1)",
    task="mlm",
    languages=("amh",),
    hub_id=None,
    requires_local=True,
    license="UNKNOWN",
    splits=("train", "validation"),
    notes=("Curated Amharic corpus for continued MLM pretraining. Supply with "
           "--local-path and record composition in the dataset card."),
)

TIGRINYA_CORPUS = DatasetSpec(
    key="tigrinya_mlm",
    name="Tigrinya monolingual corpus (Stage 1)",
    task="mlm",
    languages=("tir",),
    hub_id=None,
    requires_local=True,
    license="UNKNOWN",
    splits=("train", "validation"),
    notes=("Curated Tigrinya corpus for continued MLM pretraining. Supply with "
           "--local-path and record composition in the dataset card."),
)


REGISTRY: dict[str, DatasetSpec] = {
    s.key: s for s in (
        TIGQA, AMQA, MASAKHANER_AMH, TIGRINYA_NER, AFRISENTI,
        AMHARIC_CORPUS, TIGRINYA_CORPUS,
    )
}


def get(key: str) -> DatasetSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def by_task(task: str) -> list[DatasetSpec]:
    return [s for s in REGISTRY.values() if s.task == task]
