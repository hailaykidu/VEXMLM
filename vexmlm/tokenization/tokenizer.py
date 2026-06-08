"""VEXMLM tokenizer wrapper."""

from transformers import XLMRobertaTokenizer
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class VEXMLMTokenizer:
    """Wrapper for VEXMLM tokenizer with vocabulary expansion support."""

    def __init__(self, base_model: str = "xlm-roberta-base"):
        """
        Initialize VEXMLM tokenizer.

        Args:
            base_model: Base model name from HuggingFace
        """
        self.base_model = base_model
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(base_model)
        self.original_vocab_size = len(self.tokenizer)

    def add_vocabulary(
        self,
        new_tokens_path: Union[str, Path],
    ) -> int:
        """
        Add new tokens from vocabulary file to tokenizer.

        Args:
            new_tokens_path: Path to vocabulary JSON file or list of tokens

        Returns:
            Number of tokens added
        """
        if isinstance(new_tokens_path, (str, Path)):
            import json

            with open(new_tokens_path, "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
            new_tokens = list(vocab_dict.keys())
        else:
            new_tokens = new_tokens_path

        num_added = self.tokenizer.add_tokens(new_tokens)
        logger.info(
            f"Added {num_added} new tokens. "
            f"Tokenizer vocab size: {len(self.tokenizer)}"
        )

        return num_added

    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.tokenizer)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ):
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', etc.)
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt', 'tf', 'np', etc.)

        Returns:
            Encoded output
        """
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, output_path: Union[str, Path]) -> None:
        """
        Save tokenizer to disk.

        Args:
            output_path: Path to save tokenizer
        """
        self.tokenizer.save_pretrained(str(output_path))
        logger.info(f"Tokenizer saved to {output_path}")

    @staticmethod
    def from_pretrained(model_path: Union[str, Path]) -> "VEXMLMTokenizer":
        """
        Load tokenizer from disk.

        Args:
            model_path: Path to tokenizer

        Returns:
            VEXMLMTokenizer instance
        """
        tokenizer_obj = VEXMLMTokenizer.__new__(VEXMLMTokenizer)
        tokenizer_obj.tokenizer = XLMRobertaTokenizer.from_pretrained(str(model_path))
        tokenizer_obj.base_model = "custom"
        return tokenizer_obj
