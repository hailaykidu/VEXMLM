"""VEXMLM model class."""

import torch
from transformers import (
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
)
from pathlib import Path
from typing import Literal, Optional, Union
import logging

from .embedding_init import EmbeddingInitializer
from ..tokenization import VEXMLMTokenizer

logger = logging.getLogger(__name__)


class VEXMLM:
    """VEXMLM model wrapper for vocabulary-extended XLM-R."""

    def __init__(
        self,
        base_model: str = "xlm-roberta-base",
        embedding_init_strategy: Literal["mean", "random", "mixed"] = "mean",
    ):
        """
        Initialize VEXMLM model.

        Args:
            base_model: Base XLM-R model
            embedding_init_strategy: Strategy for initializing new embeddings
        """
        self.base_model = base_model
        self.embedding_init_strategy = embedding_init_strategy
        self.tokenizer = VEXMLMTokenizer(base_model)
        self.model = None
        self.num_new_tokens = 0

    def load_pretrained(self, task: Literal["mlm", "qa", "ner", "sa"] = "mlm"):
        """
        Load pretrained model for specific task.

        Args:
            task: Task type (mlm, qa, ner, sa)

        Returns:
            Self for chaining
        """
        if task == "mlm":
            self.model = XLMRobertaForMaskedLM.from_pretrained(self.base_model)
        elif task == "qa":
            self.model = XLMRobertaForSequenceClassification.from_pretrained(
                self.base_model, num_labels=2
            )
        elif task == "sa":
            self.model = XLMRobertaForSequenceClassification.from_pretrained(
                self.base_model, num_labels=2
            )
        elif task == "ner":
            self.model = XLMRobertaForTokenClassification.from_pretrained(
                self.base_model, num_labels=9
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        logger.info(f"Loaded {self.base_model} for task: {task}")
        return self

    def expand_vocabulary(
        self,
        new_tokens_path: Union[str, Path],
        embedding_init_strategy: Optional[str] = None,
    ):
        """
        Expand model vocabulary with new tokens.

        Args:
            new_tokens_path: Path to new tokens vocabulary
            embedding_init_strategy: Override default strategy

        Returns:
            Self for chaining
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")

        strategy = embedding_init_strategy or self.embedding_init_strategy

        # Add tokens to tokenizer
        self.num_new_tokens = self.tokenizer.add_vocabulary(new_tokens_path)

        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Initialize new embeddings
        initializer = EmbeddingInitializer(
            hidden_size=self.model.config.hidden_size,
            strategy=strategy,
        )

        new_embeddings = initializer.initialize_embeddings(
            self.model,
            self.num_new_tokens,
        )

        initializer.apply_initialization(
            self.model,
            new_embeddings,
            self.num_new_tokens,
        )

        logger.info(
            f"Expanded vocabulary: +{self.num_new_tokens} tokens "
            f"(total: {self.tokenizer.get_vocab_size()})"
        )

        return self

    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        """
        Save model and tokenizer to disk.

        Args:
            output_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model and tokenizer saved to {output_dir}")

    @staticmethod
    def from_pretrained(model_path: Union[str, Path], task: str = "mlm") -> "VEXMLM":
        """
        Load pretrained VEXMLM model from disk.

        Args:
            model_path: Path to saved model
            task: Task type

        Returns:
            VEXMLM instance
        """
        vexmlm = VEXMLM.__new__(VEXMLM)
        vexmlm.base_model = "custom"
        vexmlm.embedding_init_strategy = "mean"
        vexmlm.tokenizer = VEXMLMTokenizer.from_pretrained(model_path)
        vexmlm.num_new_tokens = 0

        model_path = Path(model_path)

        if task == "mlm":
            vexmlm.model = XLMRobertaForMaskedLM.from_pretrained(str(model_path))
        elif task == "qa":
            vexmlm.model = XLMRobertaForSequenceClassification.from_pretrained(
                str(model_path)
            )
        elif task == "ner":
            vexmlm.model = XLMRobertaForTokenClassification.from_pretrained(
                str(model_path)
            )
        elif task == "sa":
            vexmlm.model = XLMRobertaForSequenceClassification.from_pretrained(
                str(model_path)
            )

        logger.info(f"Loaded VEXMLM model from {model_path}")
        return vexmlm

    def get_model(self):
        """Get underlying HuggingFace model."""
        return self.model

    def get_tokenizer(self) -> VEXMLMTokenizer:
        """Get VEXMLM tokenizer."""
        return self.tokenizer
