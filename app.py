"""Dashboard Streamlit per lo smistamento delle recensioni hotel."""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.text_utils import combine_title_body


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="Hotel Review Router",
    page_icon="🏨",
    layout="wide",
)


@st.cache_resource
def load_models():
    department_path = MODELS_DIR / "department_model.joblib"
    sentiment_path = MODELS_DIR / "sentiment_model.joblib"
    if not department_path.exists() or not sentiment_path.exists():
        raise FileNotFoundError(
            "Modelli non trovati. Eseguire prima: python run_all.py"
        )
    return joblib.load(department_path), joblib.load(sentiment_path)


def predict_rows(df: pd.DataFrame, department_model, sentiment_model) -> pd.DataFrame:
    required = {"title", "body"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Il CSV deve contenere le colonne 'title' e 'body'.")

    result = df.copy()
    if "id" not in result.columns:
        result.insert(0, "id", [f"ROW-{i:04d}" for i in range(1, len(result) + 1)])

    texts = [
        combine_title_body(title, body)
        for title, body in zip(
            result["title"].fillna(""),
            result["body"].fillna(""),
            strict=True,
        )
    ]
    dep_pred = department_model.predict(texts)
    dep_proba = department_model.predict_proba(texts)
    sent_pred = sentiment_model.predict(texts)
    sent_proba = sentiment_model.predict_proba(texts)

    result["predicted_department"] = dep_pred
    result["department_probability"] = dep_proba.max(axis=1)
    result["predicted_sentiment"] = sent_pred
    result["sentiment_probability"] = sent_proba.max(axis=1)
    result["manual_review_recommended"] = (
        (result["department_probability"] < 0.60)
        | (result["sentiment_probability"] < 0.60)
    )
    result["prediction_timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return result


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


st.title("Smistamento recensioni hotel e analisi del sentiment")
st.caption(
    "Prototipo didattico basato su TF-IDF e Regressione Logistica. "
    "Le probabilità sono indicatori del modello, non certezze assolute."
)

try:
    department_model, sentiment_model = load_models()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Informazioni sul modello")
    metrics_path = OUTPUTS_DIR / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        st.metric("Accuracy reparto", f"{metrics['department']['accuracy']:.1%}")
        st.metric("F1 macro reparto", f"{metrics['department']['f1_macro']:.1%}")
        st.metric("Accuracy sentiment", f"{metrics['sentiment']['accuracy']:.1%}")
        st.metric("F1 macro sentiment", f"{metrics['sentiment']['f1_macro']:.1%}")
    st.info(
        "Una confidenza inferiore al 60% attiva il suggerimento di revisione manuale."
    )

single_tab, batch_tab = st.tabs(["Recensione singola", "Predizione batch CSV"])

with single_tab:
    title = st.text_input("Titolo della recensione", placeholder="Es. Check-in troppo lento")
    body = st.text_area(
        "Testo della recensione",
        height=180,
        placeholder="Incollare qui il commento dell'ospite...",
    )

    if st.button("Analizza recensione", type="primary"):
        if not title.strip() and not body.strip():
            st.warning("Inserire almeno il titolo o il testo della recensione.")
        else:
            text = [combine_title_body(title, body)]
            dep_label = department_model.predict(text)[0]
            dep_probs = department_model.predict_proba(text)[0]
            dep_classes = department_model.named_steps["classifier"].classes_
            dep_probability = float(dep_probs.max())

            sent_label = sentiment_model.predict(text)[0]
            sent_probs = sentiment_model.predict_proba(text)[0]
            sent_classes = sentiment_model.named_steps["classifier"].classes_
            sent_probability = float(sent_probs.max())

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Reparto consigliato")
                st.metric(dep_label, f"{dep_probability:.1%} di confidenza")
                st.dataframe(
                    pd.DataFrame(
                        {"reparto": dep_classes, "probabilità": dep_probs}
                    ).sort_values("probabilità", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
            with col2:
                st.subheader("Sentiment stimato")
                st.metric(sent_label.capitalize(), f"{sent_probability:.1%} di confidenza")
                st.dataframe(
                    pd.DataFrame(
                        {"sentiment": sent_classes, "probabilità": sent_probs}
                    ).sort_values("probabilità", ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )

            if dep_probability < 0.60 or sent_probability < 0.60:
                st.warning(
                    "Il caso è poco certo o potenzialmente ambiguo: è consigliata una verifica manuale."
                )
            else:
                st.success("Predizione con confidenza superiore alla soglia operativa del 60%.")

with batch_tab:
    uploaded = st.file_uploader(
        "Carica un CSV con almeno le colonne title e body",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            predictions = predict_rows(batch_df, department_model, sentiment_model)
            st.dataframe(predictions, use_container_width=True, hide_index=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Scarica risultati CSV",
                data=csv_bytes(predictions),
                file_name=f"predizioni_recensioni_{timestamp}.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:  # Interfaccia: mostra un errore leggibile all'utente.
            st.error(f"Impossibile elaborare il file: {exc}")
