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
    version: str = "unversioned"
    text_field: str | None = None
    label_field: str | None = None
    notes: str = ""
    splits: tuple[str, ...] = ("train", "validation", "test")
    requires_local: bool = False
    local_dir: str | None = None   # subdirectory under datasets/raw/
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- Question Answering ---------------------------------------------------------

TIGQA = DatasetSpec(
    key="tigqa",
    name="TIGQA — Tigrinya Question Answering",
    task="qa",
    languages=("tir",),
    hub_id=None,               # distributed via Zenodo, not the Hub
    requires_local=True,
    local_dir="tigqa",
    url="https://zenodo.org/records/11423987",
    citation=(
        "@inproceedings{teklehaymanot2024tigqa,\n"
        "  title={TIGQA: An Expert-Annotated Question-Answering Dataset "
        "in Tigrinya},\n"
        "  author={Teklehaymanot, Hailay and others},\n"
        "  year={2024},\n"
        "  note={Zenodo: https://zenodo.org/records/11423987}\n}"
    ),
    license="CC BY 4.0",   # verified on the Zenodo record
    version="TIGQA-1.0",
    notes=("SQuAD-format extractive QA, stored flat (title/context/qas, with no "
           "'paragraphs' level). 797 question-answer pairs over 365 contexts, "
           "split 80/10/10 by context with seed 42: 644/67/86 questions over "
           "292/36/37 contexts. 1,039 abstractive and 272 unanswerable items "
           "were excluded upstream, since answer_start == -1 cannot be scored "
           "by span extraction; no negative offsets remain. "
           "See docs/DATASET_PROVENANCE.md."),
)

AMQA = DatasetSpec(
    key="amqa",
    name="AmQA — Amharic Question Answering",
    task="qa",
    languages=("amh",),
    hub_id=None,
    requires_local=True,
    local_dir="amqa",
    url="https://github.com/semantic-systems/amharic-qa",
    citation=(
        "@inproceedings{taffa2024amqa,\n"
        "  title={Low-Resource Question Answering: An Amharic "
        "Benchmarking Dataset},\n"
        "  author={Taffa, Tilahun Abedissa and others},\n"
        "  year={2024},\n"
        "  note={https://github.com/semantic-systems/amharic-qa}\n}"
    ),
    license="MIT",
    version="SQuAD v2.0 format",
    notes=("Official train/dev/test splits used as released: 1,723/600/299 "
           "questions. 20 answer offsets were corrected upstream and one "
           "dict-wrapped paragraph list normalized; no questions dropped."),
)

# --- Named Entity Recognition ---------------------------------------------------

MASAKHANER_AMH = DatasetSpec(
    key="masakhaner_amh",
    name="MasakhaNER — Amharic",
    task="ner",
    languages=("amh",),
    hub_id="masakhane/masakhaner2",
    hub_config="amh",
    requires_local=True,       # a prepared local copy ships with this repo
    local_dir="masakhaner_amh",
    url="https://github.com/masakhane-io/masakhane-ner",
    citation=(
        "@inproceedings{adelani-etal-2021-masakhaner,\n"
        "  title={MasakhaNER: Named Entity Recognition for African Languages},\n"
        "  author={Adelani, David Ifeoluwa and others},\n"
        "  journal={TACL},\n  year={2021}\n}"
    ),
    license="CC BY-NC 4.0",   # NON-COMMERCIAL; verified in the upstream README
    version="MasakhaNER v1 (amh)",
    notes=("Official splits used as released: 1,750/250/500 sentences "
           "(25,819/3,749/7,449 tokens). CoNLL BIO tags: PER, ORG, LOC, DATE. "
           "LICENCE IS NON-COMMERCIAL (CC BY-NC 4.0), unlike most of the other "
           "datasets here; underlying news text carries per-site licences. Any "
           "model fine-tuned on it inherits a non-commercial constraint. "
           "See docs/DATASET_REDISTRIBUTION.md."),
)

TIGRINYA_NER = DatasetSpec(
    key="tigrinya_ner",
    name="Tigrinya NER",
    task="ner",
    languages=("tir",),
    hub_id=None,
    requires_local=True,
    local_dir="tigrinya_ner",
    url="https://github.com/mehari-eng/Tigrinya-NER",
    citation=(
        "@article{yohannes2022tigrinya,\n"
        "  title={Named Entity Recognition for Tigrinya},\n"
        "  author={Yohannes, Hailemariam Mehari and Amagasa, Toshiyuki},\n"
        "  year={2022},\n"
        "  note={https://github.com/mehari-eng/Tigrinya-NER}\n}"
    ),
    license="UNLICENSED — no licence file in the source repository",
    version="Yohannes and Amagasa (2022)",
    notes=("Tigrinya is NOT covered by MasakhaNER v1 or v2; this is a separate "
           "resource. Splits: 4,562/570/571 sentences (88,102/11,003/10,818 "
           "tokens). Tags: PER, ORG, LOC, DATE, MISC. NOTE: the train split "
           "contains one malformed tag ('B-LO' at line 1350, evidently a "
           "truncated 'B-LOC'); see docs/DATASET_PROVENANCE.md for how it is "
           "handled."),
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

_MLM_NOTE = (
    "200,000 lines, imported from a prepared monolingual corpus in the local "
    "LGSE workspace. PROVENANCE UNRESOLVED: the designated reference source is "
    "HornMT, but HornMT is a ~2,030-sentence parallel corpus and cannot be the "
    "origin of a 200,000-line monolingual file. Content sampling shows "
    "religious translations plus general web text, consistent with a "
    "CC-100/OSCAR-style crawl. Treated as a release blocker until the authors "
    "confirm the true source. See docs/DATASET_PROVENANCE.md."
)

AMHARIC_CORPUS = DatasetSpec(
    key="amharic_mlm",
    name="Amharic monolingual corpus (Stage 1)",
    task="mlm",
    languages=("amh",),
    hub_id=None,
    requires_local=True,
    local_dir="amharic",
    url="https://github.com/asmelashteka/HornMT",
    license="UNKNOWN — blocker",
    version="lgse-lapt-200k",
    splits=("train", "validation"),
    notes=_MLM_NOTE + " Amharic: 200,001 lines, 9,192,496 characters.",
)

TIGRINYA_CORPUS = DatasetSpec(
    key="tigrinya_mlm",
    name="Tigrinya monolingual corpus (Stage 1)",
    task="mlm",
    languages=("tir",),
    hub_id=None,
    requires_local=True,
    local_dir="tigrinya",
    url="https://github.com/asmelashteka/HornMT",
    license="UNKNOWN — blocker",
    version="lgse-lapt-200k",
    splits=("train", "validation"),
    notes=_MLM_NOTE + " Tigrinya: 200,000 lines, 6,982,894 characters.",
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
