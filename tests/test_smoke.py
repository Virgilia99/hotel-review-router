from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate_dataset import generate_reviews
from src.text_utils import normalize_text


def test_dataset_shape_and_balance() -> None:
    df = generate_reviews(n_samples=240, seed=7)
    assert len(df) == 240
    assert set(df.columns) == {"id", "title", "body", "department", "sentiment"}
    counts = df.groupby(["department", "sentiment"]).size()
    assert counts.nunique() == 1
    assert not df.duplicated(subset=["title", "body"]).any()


def test_normalization() -> None:
    assert normalize_text("Check-in: RAPIDO!") == "check in rapido"
    assert normalize_text("  Camera   pulita. ") == "camera pulita"
