"""Addestramento, valutazione e salvataggio dei modelli ML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_utils import combine_title_body, normalize_text


RANDOM_STATE = 42
DEPARTMENT_LABELS = ["Housekeeping", "Reception", "F&B"]
SENTIMENT_LABELS = ["negativo", "positivo"]


def build_pipeline(*, binary: bool) -> Pipeline:
    """Costruisce una pipeline TF-IDF + Regressione Logistica."""
    classifier = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear" if binary else "lbfgs",
        C=2.0,
    )
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=8_000,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def _classification_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in labels:
        rows.append(
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
        )
    rows.append(
        {
            "class": "macro avg",
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "support": int(report["macro avg"]["support"]),
        }
    )
    return pd.DataFrame(rows)


def _save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_f1_chart(
    department_report: pd.DataFrame,
    sentiment_report: pd.DataFrame,
    output_path: Path,
) -> None:
    data = pd.concat(
        [
            department_report.loc[department_report["class"] != "macro avg", ["class", "f1_score"]],
            sentiment_report.loc[sentiment_report["class"] != "macro avg", ["class", "f1_score"]],
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    bars = ax.bar(data["class"], data["f1_score"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score per classe sul test set")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, data["f1_score"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _error_rows(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    task: str,
) -> pd.DataFrame:
    result = test_df[["id", "title", "body", "department", "sentiment"]].copy()
    result["task"] = task
    result["true_label"] = y_true.to_numpy()
    result["predicted_label"] = y_pred
    result["confidence"] = probabilities.max(axis=1)
    result["correct"] = result["true_label"] == result["predicted_label"]

    # Probabilita' della classe predetta e, per trasparenza, distribuzione completa.
    for idx, class_name in enumerate(classes):
        safe_name = str(class_name).replace("&", "and").replace(" ", "_").lower()
        result[f"prob_{safe_name}"] = probabilities[:, idx]

    errors = result.loc[~result["correct"]].sort_values("confidence", ascending=False)
    if errors.empty:
        # In assenza di errori, restituisce i casi corretti con confidenza piu' bassa:
        # sono comunque utili per discutere le osservazioni borderline.
        errors = result.sort_values("confidence", ascending=True).head(10)
        errors = errors.assign(note="nessun errore; casi corretti a confidenza piu' bassa")
    else:
        errors = errors.head(20).assign(note="misclassificazione")
    return errors


def train_and_evaluate(
    data_path: Path,
    models_dir: Path,
    outputs_dir: Path,
) -> dict[str, object]:
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = {"id", "title", "body", "department", "sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Il dataset contiene valori mancanti nelle colonne obbligatorie")
    if df.duplicated(subset=["title", "body"]).any():
        raise ValueError("Il dataset contiene recensioni duplicate")

    df = df.copy()
    df["text"] = [
        combine_title_body(title, body)
        for title, body in zip(df["title"], df["body"], strict=True)
    ]
    strata = df["department"].astype(str) + "__" + df["sentiment"].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    department_model = build_pipeline(binary=False)
    sentiment_model = build_pipeline(binary=True)

    department_model.fit(train_df["text"], train_df["department"])
    sentiment_model.fit(train_df["text"], train_df["sentiment"])

    dep_pred = department_model.predict(test_df["text"])
    sent_pred = sentiment_model.predict(test_df["text"])
    dep_proba = department_model.predict_proba(test_df["text"])
    sent_proba = sentiment_model.predict_proba(test_df["text"])

    dep_accuracy = accuracy_score(test_df["department"], dep_pred)
    dep_f1 = f1_score(test_df["department"], dep_pred, average="macro")
    sent_accuracy = accuracy_score(test_df["sentiment"], sent_pred)
    sent_f1 = f1_score(test_df["sentiment"], sent_pred, average="macro")

    dep_report = _classification_table(test_df["department"], dep_pred, DEPARTMENT_LABELS)
    sent_report = _classification_table(test_df["sentiment"], sent_pred, SENTIMENT_LABELS)
    dep_report.to_csv(outputs_dir / "classification_report_department.csv", index=False)
    sent_report.to_csv(outputs_dir / "classification_report_sentiment.csv", index=False)

    _save_confusion_matrix(
        test_df["department"],
        dep_pred,
        DEPARTMENT_LABELS,
        "Matrice di confusione - classificazione reparto",
        outputs_dir / "confusion_matrix_department.png",
    )
    _save_confusion_matrix(
        test_df["sentiment"],
        sent_pred,
        SENTIMENT_LABELS,
        "Matrice di confusione - analisi del sentiment",
        outputs_dir / "confusion_matrix_sentiment.png",
    )
    _save_f1_chart(dep_report, sent_report, outputs_dir / "f1_by_class.png")

    dep_errors = _error_rows(
        test_df,
        test_df["department"],
        dep_pred,
        dep_proba,
        department_model.named_steps["classifier"].classes_,
        "department",
    )
    sent_errors = _error_rows(
        test_df,
        test_df["sentiment"],
        sent_pred,
        sent_proba,
        sentiment_model.named_steps["classifier"].classes_,
        "sentiment",
    )
    pd.concat([dep_errors, sent_errors], ignore_index=True).to_csv(
        outputs_dir / "error_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = test_df[["id", "title", "body", "department", "sentiment"]].copy()
    predictions["predicted_department"] = dep_pred
    predictions["department_confidence"] = dep_proba.max(axis=1)
    predictions["predicted_sentiment"] = sent_pred
    predictions["sentiment_confidence"] = sent_proba.max(axis=1)
    predictions.to_csv(outputs_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(department_model, models_dir / "department_model.joblib")
    joblib.dump(sentiment_model, models_dir / "sentiment_model.joblib")

    metrics: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split": "80/20 stratificato sulla combinazione reparto-sentiment",
        "department": {
            "accuracy": float(dep_accuracy),
            "f1_macro": float(dep_f1),
        },
        "sentiment": {
            "accuracy": float(sent_accuracy),
            "f1_macro": float(sent_f1),
        },
        "parameters": {
            "vectorizer": {
                "type": "TfidfVectorizer",
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.98,
                "sublinear_tf": True,
                "max_features": 8000,
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": 2.0,
                "class_weight": "balanced",
                "max_iter": 2000,
            },
        },
    }
    with (outputs_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "reviews_synthetic.csv")
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--outputs-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(args.data, args.models_dir, args.outputs_dir)


if __name__ == "__main__":
    main()
