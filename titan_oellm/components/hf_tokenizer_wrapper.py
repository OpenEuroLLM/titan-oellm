# Copyright (c) Titan-OELLM Custom Components.

"""
HuggingFace tokenizer wrapper for torchtitan v0.1.0 compatibility.
"""

import os
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HFTokenizerWrapper(Tokenizer):
    """Wrapper for HuggingFace tokenizers to work with torchtitan v0.1.0."""

    def __init__(self, tokenizer_path: str):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

        # Load the HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Set torchtitan-compatible attributes
        self._n_words = len(self.tokenizer)
        self.eos_id = self.tokenizer.eos_token_id or 0
        self.bos_id = getattr(self.tokenizer, 'bos_token_id', 0) or 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def build_hf_tokenizer(job_config: JobConfig) -> HFTokenizerWrapper:
    """Build HuggingFace tokenizer compatible with torchtitan v0.1.0."""
    tokenizer_path = job_config.model.tokenizer_path
    return HFTokenizerWrapper(tokenizer_path)