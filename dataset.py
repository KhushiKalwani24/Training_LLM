# dataset.py
# ─────────────────────────────────────────────────────────────
# Handles reading raw text, tokenizing it, and packaging it
# into PyTorch Dataset objects the DataLoader can iterate over.
# ─────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """
    Custom Dataset for language model training.

    Reads a plain .txt file, tokenizes it, then splits it into
    fixed-length chunks. Each chunk becomes one training sample.

    Args:
        file_path  (str): Path to the .txt training file
        tokenizer       : HuggingFace tokenizer instance
        max_length (int): Token length per training chunk
    """

    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ── Step 1: Read raw text ──────────────────────────
        print(f"[Dataset] Loading data from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # ── Step 2: Tokenize entire corpus at once ─────────
        # We don't pad/truncate yet — we'll chunk manually below
        print("[Dataset] Tokenizing...")
        tokenized = tokenizer(
            raw_text,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=False,  # Don't cut off — we handle chunking
            padding=False,
        )
        # token_ids shape: [1, total_tokens]
        token_ids = tokenized["input_ids"].squeeze()  # → [total_tokens]

        # ── Step 3: Split into fixed-length chunks ─────────
        # Each chunk = one training example
        # Drop the last incomplete chunk to keep shapes uniform
        total_tokens = token_ids.size(0)
        num_chunks = total_tokens // max_length  # Integer division

        # Trim to exactly num_chunks * max_length tokens
        token_ids = token_ids[:num_chunks * max_length]

        # Reshape into [num_chunks, max_length]
        self.examples = token_ids.view(num_chunks, max_length)

        print(f"[Dataset] Created {num_chunks} training chunks "
              f"of {max_length} tokens each.")

    def __len__(self):
        """Returns total number of training samples."""
        return self.examples.size(0)

    def __getitem__(self, idx):
        """
        Returns one training sample as a dict.

        For causal LM (next-token prediction):
          - input_ids  = tokens [0 … N-1]
          - labels     = tokens [0 … N-1]  ← same!
            HuggingFace models internally shift labels by 1
        """
        chunk = self.examples[idx]  # Shape: [max_length]
        return {
            "input_ids": chunk,
            "labels": chunk.clone(),  # Clone so gradients don't interfere
        }


def build_tokenizer_and_dataset(config):
    """
    Convenience function: loads tokenizer + builds dataset in one call.

    Returns:
        tokenizer  : HuggingFace tokenizer (needed for inference later)
        dataset    : TextDataset instance ready for DataLoader
    """

    # ── Load pretrained tokenizer ──────────────────────────
    print(f"[Tokenizer] Loading '{config.MODEL_NAME}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # GPT-2 has no pad token by default — set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Tokenizer] Set pad_token = eos_token")

    # ── Build dataset ──────────────────────────────────────
    dataset = TextDataset(
        file_path=config.DATA_PATH,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH,
    )

    return tokenizer, dataset