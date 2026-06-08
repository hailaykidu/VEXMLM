"""Constants and configuration for VEXMLM."""

# Ge'ez-script languages (primary focus)
GEZ_LANGUAGES = {
    "am": "Amharic",
    "ti": "Tigrinya",
}

# Extended to 19 African low-resource languages
LANGUAGES = {
    # Ge'ez script
    "am": "Amharic",
    "ti": "Tigrinya",
    # Bantu languages
    "sw": "Swahili",
    "zu": "Zulu",
    "xh": "Xhosa",
    "st": "Sotho",
    "sn": "Shona",
    # Afro-Asiatic
    "ha": "Hausa",
    "om": "Oromo",
    "so": "Somali",
    # Niger-Congo
    "yo": "Yoruba",
    "ig": "Igbo",
    "ff": "Fulani",
    # Additional African languages
    "ln": "Lingala",
    "rw": "Kinyarwanda",
    "ny": "Nyanja",
    "to": "Tonga",
    "ki": "Kikuyu",
    "mg": "Malagasy",
}

# Task types
TASK_TYPES = {
    "qa": "Question Answering",
    "ner": "Named Entity Recognition",
    "sa": "Sentiment Analysis",
}

# Evaluation metrics by task
TASK_METRICS = {
    "qa": ["exact_match", "f1"],
    "ner": ["precision", "recall", "f1", "oov_accuracy"],
    "sa": ["accuracy", "f1"],
}

# XLM-R base configuration
XLM_R_CONFIG = {
    "model_name": "xlm-roberta-base",
    "vocab_size": 250002,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
}

# VEXMLM expansion parameters
VEXMLM_EXPANSION = {
    "new_vocab_size": 30000,
    "embedding_init_strategy": "mean",
    "total_vocab_size": 280002,
}

# Training defaults
TRAINING_DEFAULTS = {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "max_seq_length": 512,
    "seed": 42,
}

# Pretraining defaults
PRETRAINING_DEFAULTS = {
    "mlm_probability": 0.15,
    "num_epochs": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
}
