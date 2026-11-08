"""Language identification with fastText lid.176 (plan 9.1), or unknown when it is absent.

Unknown languages use the global graph prior (plan 6.2), so a missing model only means
slightly less specific scoring.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

UNK = "<unk>"
LID_TO_LANG = {
    "af": "afr", "am": "amh", "ar": "arq", "de": "deu", "en": "eng", "es": "esp", "ha": "hau", "hi": "hin", "ig": "ibo",
    "id": "ind", "jv": "jav", "rw": "kin", "mr": "mar", "om": "orm", "pt": "ptbr", "ro": "ron", "ru": "rus", "so": "som",
    "su": "sun", "sw": "swa", "sv": "swe", "tt": "tat", "ti": "tir", "uk": "ukr", "xh": "xho", "yo": "yor", "zh": "chn",
    "zu": "zul",
}
_model = None


def _load(path: str | Path):
    global _model
    if _model is None:
        import fasttext

        fasttext.FastText.eprint = lambda *_: None  # silences a deprecation print
        _model = fasttext.load_model(str(path))
    return _model


def detect(texts: Sequence[str], model_path: str | Path, min_prob: float = 0.4) -> list[str]:
    """BRIGHTER language codes; anything uncertain or unmapped is <unk>."""
    if not os.path.exists(model_path):
        return [UNK] * len(texts)
    model = _load(model_path)
    out = []
    for t in texts:
        # Calling the C++ binding avoids fasttext's predict(), which breaks on numpy 2.
        pred = model.f.predict(" ".join(str(t).split()) + "\n", 1, 0.0, "strict")
        if not pred:
            out.append(UNK)
            continue
        prob, label = pred[0]
        code = label.replace("__label__", "")
        out.append(LID_TO_LANG.get(code, UNK) if np.isfinite(prob) and prob >= min_prob else UNK)
    return out
