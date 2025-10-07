"""
Text datasets for LLM experiments.

Provides dataset loaders for common text datasets used in language model
evaluation and pruning experiments.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Generic text dataset wrapper.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        return_tensors: Whether to return tensors (default: True)
    """

    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 512, return_tensors: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt" if self.return_tensors else None
        )

        if self.return_tensors:
            # Squeeze batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding


class WikiTextDataset(Dataset):
    """
    WikiText dataset for language modeling.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation', 'test')
        max_length: Maximum sequence length
        dataset_name: Specific WikiText version
    """

    def __init__(self, tokenizer: Any, split: str = "test", max_length: int = 512, dataset_name: str = "wikitext-2-raw-v1"):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading WikiText dataset: {dataset_name} ({split})")
        self.dataset = load_dataset("wikitext", dataset_name, split=split)

        # Filter out empty texts
        self.texts = [item["text"] for item in self.dataset if item["text"] and len(item["text"].strip()) > 0]

        logger.info(f"Loaded {len(self.texts)} text samples")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Squeeze batch dimension and add labels for LM
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()

        return encoding


class C4Dataset(IterableDataset):
    """
    C4 (Colossal Clean Crawled Corpus) dataset.

    This is a streaming dataset due to its large size.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to load (None = all)
    """

    def __init__(self, tokenizer: Any, split: str = "validation", max_length: int = 512, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

        from datasets import load_dataset

        logger.info(f"Loading C4 dataset ({split}, streaming)")
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0

        for item in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break

            text = item["text"]
            if not text or len(text.strip()) == 0:
                continue

            encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            encoding["labels"] = encoding["input_ids"].clone()

            yield encoding
            count += 1


def load_text_dataset(
    dataset_name: str, tokenizer: Any, split: str = "test", max_length: int = 512, max_samples: Optional[int] = None, **kwargs
) -> Dataset:
    """
    Load a text dataset by name.

    Args:
        dataset_name: Name of dataset ('wikitext', 'c4', 'ptb')
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        max_length: Maximum sequence length
        max_samples: Optional sample limit
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "wikitext":
        return WikiTextDataset(tokenizer, split, max_length, **kwargs)

    elif dataset_name == "c4":
        return C4Dataset(tokenizer, split, max_length, max_samples)

    elif dataset_name == "ptb":
        from datasets import load_dataset

        logger.info(f"Loading PTB dataset ({split})")
        dataset = load_dataset("ptb_text_only", split=split)
        texts = [item["sentence"] for item in dataset if item["sentence"]]
        if max_samples:
            texts = texts[:max_samples]
        return TextDataset(texts, tokenizer, max_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. " f"Supported: wikitext, c4, ptb")


# Register datasets in alignment registry if needed
try:
    from ...core.registry import register_dataset

    @register_dataset("wikitext")
    def create_wikitext(**kwargs):
        """Create WikiText dataset from config."""
        return WikiTextDataset(**kwargs)

    @register_dataset("c4")
    def create_c4(**kwargs):
        """Create C4 dataset from config."""
        return C4Dataset(**kwargs)

except ImportError:
    pass
