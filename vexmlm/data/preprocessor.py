"""Data preprocessing utilities for VEXMLM."""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess text data for VEXMLM."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove control characters
        text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r"\S+@\S+", "", text)

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    @staticmethod
    def preprocess_corpus(
        texts: List[str],
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize: bool = True,
    ) -> List[str]:
        """Preprocess a corpus of texts."""
        logger.info(f"Preprocessing {len(texts)} texts...")

        processed = []
        for text in texts:
            if remove_urls:
                text = DataPreprocessor.remove_urls(text)
            if remove_emails:
                text = DataPreprocessor.remove_emails(text)
            if normalize:
                text = DataPreprocessor.normalize_whitespace(text)

            text = DataPreprocessor.clean_text(text)
            processed.append(text)

        logger.info(f"Preprocessing complete")
        return processed

    @staticmethod
    def split_sentences(text: str, lang_code: str = "en") -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on periods, question marks, exclamation marks
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
