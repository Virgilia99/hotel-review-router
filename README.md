# Hotel Review Router

Project work universitario dedicato allo **smistamento automatico delle recensioni alberghiere**
e alla **sentiment analysis** mediante Machine Learning.

## Funzionalità

- Generazione di un dataset sintetico bilanciato.
- Classificazione del reparto: Housekeeping, Reception, Food & Beverage.
- Classificazione del sentiment: positivo o negativo.
- TF-IDF + Regressione Logistica.
- Valutazione con accuracy, F1-score e matrici di confusione.
- Predizione batch da file CSV.
- Dashboard Streamlit.
- Revisione manuale suggerita quando la confidenza è inferiore al 60%.

## Struttura

```text
hotel-review-router/
├── app.py
├── run_all.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── text_utils.py
│   ├── train_models.py
│   └── predict_batch.py
└── tests/
    └── test_smoke.py
```

## Installazione

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Esecuzione completa

```bash
python run_all.py
```

## Avvio dashboard

```bash
streamlit run app.py
```

## Test

```bash
python -m pytest -q
```

## Nota

Il dataset è completamente sintetico e non contiene recensioni reali o dati personali.
Il progetto è destinato a finalità didattiche e dimostrative.
