"""
VEXMLM: Vocabulary-Extended XLM-R for Ge'ez-script Languages

A vocabulary-extended variant of XLM-R optimized for low-resource African languages,
particularly Ge'ez-script languages (Amharic and Tigrinya).

Main features:
- SentencePiece-based vocabulary expansion (30k Ge'ez tokens)
- Mean initialization strategy for new embeddings
- Two-stage training: continued MLM + task-specific finetuning
- Support for 19 African languages
- Comprehensive intrinsic and extrinsic evaluation
"""

__version__ = "1.0.0"
__author__ = "VEXMLM Team"

from .modeling import VEXMLM
from .tokenization import VEXMLMTokenizer
from .training import PretrainTrainer, FinetuneTrainer
from .evaluation import IntrinsicEvaluator, ExtrinsicEvaluator

__all__ = [
    "VEXMLM",
    "VEXMLMTokenizer",
    "PretrainTrainer",
    "FinetuneTrainer",
    "IntrinsicEvaluator",
    "ExtrinsicEvaluator",
]
