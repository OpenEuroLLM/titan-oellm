"""Shared EOS-injection helper used by every dataloader / sequencer.

The contract: callers pass `eos_id` once at construction; injection happens at
the per-document boundary in either the dataloader (DPD/BFP/ChunkedMMap/MMap)
or the sequencer (StreamingSequencer).  The helper avoids double-EOS when the
upstream `.bin` already terminates documents with the eos token.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list]


def doc_ends_with_eos(tokens: ArrayLike, eos_id: Optional[int]) -> bool:
    """Return True if `tokens` is non-empty and its last token equals `eos_id`."""
    if eos_id is None:
        return False
    if isinstance(tokens, np.ndarray):
        if tokens.size == 0:
            return False
        return int(tokens[-1]) == int(eos_id)
    if not tokens:
        return False
    return int(tokens[-1]) == int(eos_id)


def append_eos_if_missing(
    tokens: ArrayLike,
    eos_id: Optional[int],
    dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, bool]:
    """Append `eos_id` to `tokens` unless it is already present at the end.

    Returns `(out_tokens, was_appended)`.  When `eos_id is None` this is a
    no-op and the input is returned (cast to ndarray).
    """
    if isinstance(tokens, np.ndarray):
        out_dtype = dtype if dtype is not None else tokens.dtype
        arr = tokens
    else:
        out_dtype = dtype if dtype is not None else np.int64
        arr = np.asarray(tokens, dtype=out_dtype)

    if eos_id is None:
        return arr.astype(out_dtype, copy=False), False

    if doc_ends_with_eos(arr, eos_id):
        return arr.astype(out_dtype, copy=False), False

    eos_arr = np.array([eos_id], dtype=out_dtype)
    return np.concatenate([arr.astype(out_dtype, copy=False), eos_arr]), True


def eos_overhead_for_doc(
    last_token: Optional[int],
    eos_id: Optional[int],
) -> int:
    """How many extra tokens does EOS injection add for a doc whose last raw
    token equals `last_token`?  Used by length-accounting code in BFP/DPD that
    needs to plan packed-sequence sizes before reading every token.
    """
    if eos_id is None or last_token is None:
        return 0
    return 0 if int(last_token) == int(eos_id) else 1
