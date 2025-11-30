# local_tokenizer.py
# -*- coding: utf-8 -*-
"""
Local tokenizer loading module:
- Loads the AutoTokenizer only once.
- Used for token counting (smart chunking, metadata compression, etc.).
- Does not participate in model inference; inference still goes through the remote server.
"""

import logging
from typing import Optional

from transformers import AutoTokenizer

_cached_tokenizer = None


def load_local_tokenizer(
    model_path: str,
    model_max_length: Optional[int] = None,
):
    global _cached_tokenizer
    if _cached_tokenizer is not None:
        return _cached_tokenizer

    logging.info(f"[local_tokenizer] Loading tokenizer from local path: {model_path}")
    kwargs = {
        "trust_remote_code": True,
    }
    if model_max_length is not None:
        kwargs["model_max_length"] = model_max_length

    tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
    _cached_tokenizer = tok
    return tok
