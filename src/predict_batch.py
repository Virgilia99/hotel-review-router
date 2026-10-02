"""Predizione batch di reparto e sentiment da un file CSV."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.text_utils import combine_title_body


def predict_batch(
    input_path: Path,
    output_path: Path,
    department_model_path: Path,
    sentiment_model_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"title", "body"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Il CSV deve contenere le colonne: {sorted(required)}")

    if "id" not in df.columns:
        df.insert(0, "id", [f"ROW-{i:04d}" for i in range(1, len(df) + 1)])

    department_model = joblib.load(department_model_path)
    sentiment_model = joblib.load(sentiment_model_path)
    text = [
        combine_title_body(title, body)
        for title, body in zip(df["title"].fillna(""), df["body"].fillna(""), strict=True)
    ]

    department_pred = department_model.predict(text)
    department_proba = department_model.predict_proba(text)
    sentiment_pred = sentiment_model.predict(text)
    sentiment_proba = sentiment_model.predict_proba(text)

    result = df.copy()
    result["predicted_department"] = department_pred
    result["department_probability"] = department_proba.max(axis=1)
    result["predicted_sentiment"] = sentiment_pred
    result["sentiment_probability"] = sentiment_proba.max(axis=1)
    result["manual_review_recommended"] = (
        (result["department_probability"] < 0.60)
        | (result["sentiment_probability"] < 0.60)
    )
    result["prediction_timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data" / "batch_example.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / f"batch_predictions_{timestamp}.csv",
    )
    parser.add_argument(
        "--department-model",
        type=Path,
        default=root / "models" / "department_model.joblib",
    )
    parser.add_argument(
        "--sentiment-model",
        type=Path,
        default=root / "models" / "sentiment_model.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_batch(
        args.input,
        args.output,
        args.department_model,
        args.sentiment_model,
    )
    print(f"Predizioni salvate in: {args.output}")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
