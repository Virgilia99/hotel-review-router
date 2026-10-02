"""Esegue l'intero flusso: generazione dati, training e batch di esempio."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.generate_dataset import default_example_batch, export_dataset, generate_reviews
from src.predict_batch import predict_batch
from src.train_models import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "reviews_synthetic.csv"
    batch_path = root / "data" / "batch_example.csv"
    models_dir = root / "models"
    outputs_dir = root / "outputs"

    df = generate_reviews(n_samples=360, seed=42)
    export_dataset(df, data_path)
    default_example_batch().to_csv(batch_path, index=False, encoding="utf-8-sig")
    print(f"1/3 Dataset generato: {data_path}")

    train_and_evaluate(data_path, models_dir, outputs_dir)
    print("2/3 Modelli addestrati e valutati")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = outputs_dir / f"batch_predictions_{timestamp}.csv"
    predict_batch(
        batch_path,
        output_path,
        models_dir / "department_model.joblib",
        models_dir / "sentiment_model.joblib",
    )
    print(f"3/3 Predizione batch completata: {output_path}")


if __name__ == "__main__":
    main()
