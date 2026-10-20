"""Text normalization (SRS FR-1.2)."""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Unicode NFC, collapsed spaces, at most one blank line, punctuation runs capped at three (FR-1.2).

    Paragraph breaks are kept: style and structure signals need them.
    """
    t = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u00a0\u3000]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"([!?.~…。,*\-])\1{3,}", r"\1\1\1", t)
    return t.strip()
