"""
Text datasets for LLM experiments.

Provides dataset loaders for common text datasets used in language model
evaluation and pruning experiments.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from nodelens.core.registry import register_dataset

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
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        # Accept either a tokenizer object or a model ID string
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            hf_tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise TypeError(f"tokenizer must be a string or PreTrainedTokenizerBase, got {type(tokenizer)}")

        # If no pad token exists, set it to the eos token (common for causal LM)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        self.tokenizer = hf_tokenizer
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
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        # Accept either a tokenizer object or a model ID string
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            hf_tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise TypeError(f"tokenizer must be a string or PreTrainedTokenizerBase, got {type(tokenizer)}")

        # If no pad token exists, set it to the eos token (common for causal LM)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        self.tokenizer = hf_tokenizer
        self.max_length = max_length
        self.max_samples = max_samples

        from datasets import load_dataset

        logger.info(f"Loading C4 dataset ({split}, streaming)")
        self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)

        # For LLM calibration/analysis we often need raw texts (e.g., to build a reusable
        # calibration set). If max_samples is specified, materialize that many texts so
        # downstream code can rely on a `.texts` attribute (like WikiText).
        self.texts: Optional[List[str]] = None
        if self.max_samples is not None:
            texts: List[str] = []
            for item in self.dataset:
                text = item.get("text")
                if not text or len(text.strip()) == 0:
                    continue
                texts.append(text)
                if len(texts) >= self.max_samples:
                    break
            self.texts = texts
            logger.info(f"Materialized {len(self.texts)} C4 texts for split='{split}'")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0

        # If we materialized texts, iterate over them for deterministic reuse.
        iterable = self.texts if self.texts is not None else self.dataset

        for item in iterable:
            if self.max_samples and count >= self.max_samples:
                break

            text = item if isinstance(item, str) else item.get("text")
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

    elif dataset_name in {"mixed_wikitext_c4", "mixed_wiki_c4", "mixed"}:
        # Lightweight "mixed" calibration set: combine WikiText + C4 raw texts.
        # This is useful for robustness/sensitivity experiments.
        #
        # Supported kwargs:
        # - wikitext_name: which WikiText subset to use (default: wikitext-2-raw-v1)
        # - wikitext_fraction: fraction of samples drawn from WikiText when max_samples is set (default: 0.5)
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            hf_tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise TypeError(f"tokenizer must be a string or PreTrainedTokenizerBase, got {type(tokenizer)}")

        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        wikitext_name = kwargs.get("wikitext_name", "wikitext-2-raw-v1")
        wikitext_fraction = float(kwargs.get("wikitext_fraction", 0.5))

        if max_samples is None:
            # Default to a small mixed set if caller didn't specify a budget.
            max_samples = 512

        n_wiki = int(round(max_samples * wikitext_fraction))
        n_c4 = max_samples - n_wiki

        wiki_ds = WikiTextDataset(hf_tokenizer, split=split, max_length=max_length, dataset_name=wikitext_name)
        wiki_texts = list(getattr(wiki_ds, "texts", []))[:n_wiki]

        c4_ds = C4Dataset(hf_tokenizer, split=split, max_length=max_length, max_samples=n_c4)
        c4_texts = list(getattr(c4_ds, "texts", []))[:n_c4]

        # Interleave for better mixing.
        mixed_texts: List[str] = []
        for i in range(max(len(wiki_texts), len(c4_texts))):
            if i < len(wiki_texts):
                mixed_texts.append(wiki_texts[i])
            if i < len(c4_texts):
                mixed_texts.append(c4_texts[i])

        return TextDataset(mixed_texts, hf_tokenizer, max_length=max_length)

    elif dataset_name == "ptb":
        from datasets import load_dataset

        logger.info(f"Loading PTB dataset ({split})")
        dataset = load_dataset("ptb_text_only", split=split)
        texts = [item["sentence"] for item in dataset if item["sentence"]]
        if max_samples:
            texts = texts[:max_samples]
        return TextDataset(texts, tokenizer, max_length)

    elif dataset_name in {"arxiv", "scientific", "scientific_papers", "scientific_arxiv"}:
        # Scientific Papers (ArXiv) - long-form scientific text.
        from datasets import load_dataset
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            hf_tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise TypeError(f"tokenizer must be a string or PreTrainedTokenizerBase, got {type(tokenizer)}")

        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        logger.info(f"Loading scientific_papers (arxiv) dataset ({split})")
        # `scientific_papers` uses custom dataset code on HuggingFace.
        dataset = load_dataset("scientific_papers", "arxiv", split=split, trust_remote_code=True)
        texts: List[str] = []
        for item in dataset:
            t = item.get("article") or item.get("text") or item.get("abstract")
            if not t or len(str(t).strip()) == 0:
                continue
            texts.append(str(t))
            if max_samples and len(texts) >= int(max_samples):
                break
        return TextDataset(texts, hf_tokenizer, max_length=max_length)

    elif dataset_name in {"code", "code_search_net", "codesearchnet", "code-search-net"}:
        # CodeSearchNet (python) - code-heavy calibration domain.
        from datasets import load_dataset
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            hf_tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            raise TypeError(f"tokenizer must be a string or PreTrainedTokenizerBase, got {type(tokenizer)}")

        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        language = str(kwargs.get("language", "python"))
        logger.info(f"Loading code_search_net ({language}) dataset ({split})")
        # `code_search_net` uses custom dataset code on HuggingFace.
        dataset = load_dataset("code_search_net", language, split=split, trust_remote_code=True)
        texts: List[str] = []
        for item in dataset:
            t = item.get("code") or item.get("func_code_string") or item.get("content")
            if not t or len(str(t).strip()) == 0:
                continue
            texts.append(str(t))
            if max_samples and len(texts) >= int(max_samples):
                break
        return TextDataset(texts, hf_tokenizer, max_length=max_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext, c4, ptb, mixed_wikitext_c4, arxiv, code")


# Register datasets in the alignment registry.
@register_dataset("wikitext-2-v1")
def create_wikitext(**kwargs):
    """Create a WikiText dataset from config."""
    return WikiTextDataset(**kwargs)


@register_dataset("c4")
def create_c4(**kwargs):
    """Create a C4 dataset from config."""
    return C4Dataset(**kwargs)
