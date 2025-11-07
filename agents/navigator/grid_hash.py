"""Compact grid hashing utilities for navigator modules.

Mask semantics:
  - `FrameMask` is a sequence of rectangles `(y0, y1, x0, x1)`.
  - When hashing, masked regions are zeroed out so UI (e.g., energy HUD)
    rendering does not affect state identity.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence, Tuple

import numpy as np

from .types import Frame, FrameHash

MaskRect = Tuple[int, int, int, int]
FrameMask = Sequence[MaskRect]

HASH_ID_CHARS = 6


def hash_frame(frame: Frame, *, mask: Optional[FrameMask] = None) -> FrameHash:
    if not frame:
        return FrameHash("0" * HASH_ID_CHARS)

    arr = np.array(frame, dtype=np.uint8)
    if mask is not None:
        h, w = arr.shape[0], arr.shape[1]
        for y0, y1, x0, x1 in mask:
            arr[max(0, y0) : min(h, y1 + 1), max(0, x0) : min(w, x1 + 1)] = 0

    digest = hashlib.blake2s(arr.tobytes(), digest_size=max(4, (HASH_ID_CHARS + 1) // 2)).hexdigest()
    return FrameHash(digest[:HASH_ID_CHARS])
