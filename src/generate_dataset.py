"""Generatore di recensioni sintetiche per strutture ricettive.

Il dataset e' completamente artificiale e non contiene dati personali.
Le etichette sono bilanciate per reparto e sentiment. Una quota dei testi
include riferimenti a piu' reparti per introdurre casi realistici e ambigui.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


DEPARTMENTS = ("Housekeeping", "Reception", "F&B")
SENTIMENTS = ("positivo", "negativo")

SPECIFIC_TITLES: dict[str, dict[str, list[str]]] = {
    "Housekeeping": {
        "positivo": [
            "Camera impeccabile",
            "Pulizia eccellente",
            "Bagno perfetto",
            "Biancheria fresca",
            "Riassetto molto curato",
            "Ottimo servizio ai piani",
            "Ambienti davvero ordinati",
            "Asciugamani sempre puliti",
        ],
        "negativo": [
            "Camera poco pulita",
            "Bagno da rivedere",
            "Biancheria non adeguata",
            "Riassetto insufficiente",
            "Polvere in camera",
            "Problema con gli asciugamani",
            "Pulizia deludente",
            "Servizio ai piani lento",
        ],
    },
    "Reception": {
        "positivo": [
            "Check-in rapido",
            "Accoglienza eccellente",
            "Reception disponibile",
            "Pagamento semplice",
            "Check-out efficiente",
            "Staff molto professionale",
            "Prenotazione gestita bene",
            "Informazioni chiare",
        ],
        "negativo": [
            "Attesa al check-in",
            "Errore nel conto",
            "Reception poco disponibile",
            "Problema con la prenotazione",
            "Check-out lento",
            "Informazioni poco chiare",
            "Chiave elettronica non funzionante",
            "Accoglienza da migliorare",
        ],
    },
    "F&B": {
        "positivo": [
            "Colazione ottima",
            "Ristorante eccellente",
            "Servizio sala impeccabile",
            "Buffet ben assortito",
            "Cena molto buona",
            "Opzioni alimentari curate",
            "Caffè e dolci eccellenti",
            "Personale di sala attento",
        ],
        "negativo": [
            "Colazione deludente",
            "Servizio lento al ristorante",
            "Buffet scarso",
            "Cibo servito freddo",
            "Poche opzioni disponibili",
            "Cena da rivedere",
            "Caffè di bassa qualità",
            "Sala poco organizzata",
        ],
    },
}

GENERIC_TITLES: dict[str, list[str]] = {
    "positivo": [
        "Esperienza positiva",
        "Servizio molto apprezzato",
        "Soggiorno piacevole",
        "Un dettaglio che fa la differenza",
        "Buona organizzazione",
        "Tutto gestito con cura",
    ],
    "negativo": [
        "Esperienza da migliorare",
        "Un problema durante il soggiorno",
        "Servizio sotto le aspettative",
        "Organizzazione da rivedere",
        "Aspetto poco soddisfacente",
        "Disagio non risolto",
    ],
}

NEUTRAL_TITLES = [
    "Il mio soggiorno",
    "Osservazioni sul servizio",
    "Esperienza in struttura",
    "Una nota sull'organizzazione",
    "Valutazione del soggiorno",
    "Commento dopo la partenza",
]

BASE_SENTENCES: dict[str, dict[str, list[str]]] = {
    "Housekeeping": {
        "positivo": [
            "la camera era pulita, ordinata e priva di polvere",
            "il bagno è stato trovato igienizzato e ben mantenuto",
            "la biancheria era fresca e sistemata con attenzione",
            "gli asciugamani sono stati sostituiti puntualmente",
            "il riassetto quotidiano è stato preciso e discreto",
            "il pavimento e le superfici risultavano ben puliti",
            "il servizio ai piani è intervenuto con rapidità",
            "armadio e comodini erano ordinati e senza residui",
            "la doccia era pulita e perfettamente utilizzabile",
            "la stanza è stata sistemata con cura ogni giorno",
            "la richiesta di pulizia aggiuntiva è stata gestita subito",
            "l'ambiente aveva un aspetto curato e accogliente",
        ],
        "negativo": [
            "la camera presentava polvere e superfici non pulite",
            "nel bagno erano visibili residui e segni di scarsa igiene",
            "la biancheria non appariva adeguatamente pulita",
            "gli asciugamani erano macchiati e non sono stati cambiati subito",
            "il riassetto quotidiano non è stato effettuato nei tempi indicati",
            "il pavimento era sporco al rientro in stanza",
            "il servizio ai piani ha richiesto più solleciti",
            "armadio e comodini mostravano polvere evidente",
            "la doccia non era stata pulita con sufficiente attenzione",
            "la stanza è rimasta disordinata dopo il passaggio del personale",
            "la seconda richiesta di pulizia non ha risolto il problema",
            "lo standard di igiene è risultato inferiore alle aspettative",
        ],
    },
    "Reception": {
        "positivo": [
            "il check-in è stato rapido e senza complicazioni",
            "il check-out si è concluso in pochi minuti",
            "la prenotazione è stata gestita con precisione",
            "il pagamento finale è risultato chiaro e corretto",
            "la fattura è stata preparata senza errori",
            "la chiave elettronica è stata consegnata e attivata subito",
            "il deposito bagagli è stato organizzato con disponibilità",
            "la reception ha fornito una risposta cortese e risolutiva",
            "la modifica delle date è stata effettuata senza difficoltà",
            "le informazioni sui servizi erano complete e comprensibili",
            "il personale ha gestito bene anche un momento affollato",
            "la richiesta dell'ospite è stata presa in carico tempestivamente",
        ],
        "negativo": [
            "il check-in ha richiesto un'attesa eccessiva",
            "il check-out è stato rallentato da passaggi poco organizzati",
            "la prenotazione conteneva dati non aggiornati",
            "il pagamento finale includeva un addebito non chiaro",
            "la fattura è stata consegnata con un errore",
            "la chiave elettronica non funzionava e la sostituzione è stata lenta",
            "il deposito bagagli non è stato gestito come concordato",
            "la reception ha fornito indicazioni contraddittorie",
            "la modifica delle date ha richiesto troppi solleciti",
            "le informazioni sui servizi erano incomplete",
            "il personale non ha proposto una soluzione tempestiva",
            "la richiesta dell'ospite è rimasta senza risposta per troppo tempo",
        ],
    },
    "F&B": {
        "positivo": [
            "la colazione era varia, fresca e ben presentata",
            "il buffet del mattino era rifornito con continuità",
            "il servizio al ristorante è stato attento e professionale",
            "la cena è stata servita nei tempi giusti",
            "il caffè era caldo e di buona qualità",
            "la scelta dei dolci era ampia e curata",
            "le opzioni vegetariane erano chiaramente indicate",
            "la richiesta per un'intolleranza è stata gestita con attenzione",
            "il personale di sala è stato disponibile e preciso",
            "i piatti erano gustosi e ben presentati",
            "le porzioni erano adeguate e il servizio ben coordinato",
            "l'organizzazione della sala ha reso il pasto piacevole",
        ],
        "negativo": [
            "la colazione era poco varia e diversi prodotti erano terminati",
            "il buffet del mattino non è stato rifornito in tempo",
            "il servizio al ristorante è stato lento e disorganizzato",
            "la cena è arrivata fredda dopo una lunga attesa",
            "il caffè era tiepido e di qualità modesta",
            "la scelta dei dolci era limitata e poco fresca",
            "le opzioni vegetariane erano insufficienti",
            "la richiesta per un'intolleranza non è stata gestita correttamente",
            "il personale di sala ha impiegato troppo tempo a intervenire",
            "i piatti erano poco curati e inferiori alle aspettative",
            "le porzioni erano ridotte rispetto a quanto indicato",
            "l'organizzazione della sala ha causato attese evitabili",
        ],
    },
}

CONTEXTS = [
    "durante un soggiorno di due notti",
    "nel fine settimana",
    "in un momento di alta affluenza",
    "all'arrivo in struttura",
    "prima della partenza",
    "durante il secondo giorno",
    "nonostante l'hotel fosse quasi completo",
    "in occasione di un breve viaggio di lavoro",
    "durante una vacanza in coppia",
    "nel corso del soggiorno",
]

CLOSINGS: dict[str, list[str]] = {
    "positivo": [
        "Il servizio ha contribuito a una valutazione complessivamente positiva.",
        "È un aspetto che consiglierei di mantenere.",
        "La gestione è stata coerente con le aspettative.",
        "Il risultato è stato soddisfacente e ben organizzato.",
        "Questo dettaglio ha migliorato l'esperienza complessiva.",
    ],
    "negativo": [
        "Il problema dovrebbe essere preso in carico con maggiore rapidità.",
        "È un aspetto che ha inciso negativamente sul giudizio finale.",
        "Sarebbe utile una procedura più chiara e tempestiva.",
        "La situazione richiede un intervento organizzativo.",
        "Il disagio non è stato compensato durante il soggiorno.",
    ],
}


def _sentence_for(department: str, sentiment: str, rng: random.Random) -> str:
    base = rng.choice(BASE_SENTENCES[department][sentiment])
    context = rng.choice(CONTEXTS)
    if sentiment == "positivo":
        variants = [
            f"{base.capitalize()} {context}.",
            f"{context.capitalize()}, {base}.",
            f"Ho apprezzato questo aspetto: {base} {context}.",
            f"L'aspetto più positivo è stato questo: {base}.",
        ]
    else:
        variants = [
            f"{base.capitalize()} {context}.",
            f"{context.capitalize()}, {base}.",
            f"Ho riscontrato questo problema: {base} {context}.",
            f"Il principale problema è stato questo: {base}.",
        ]
    return rng.choice(variants)


def _strip_period(sentence: str) -> str:
    return sentence[:-1] if sentence.endswith(".") else sentence


def _ambiguous_body(target_department: str, sentiment: str, rng: random.Random) -> str:
    other_department = rng.choice([d for d in DEPARTMENTS if d != target_department])
    opposite = "negativo" if sentiment == "positivo" else "positivo"
    target_sentence = _strip_period(_sentence_for(target_department, sentiment, rng))
    other_sentence = _strip_period(_sentence_for(other_department, opposite, rng))

    # La seconda clausola, introdotta da "ma", rappresenta il tema prevalente
    # e determina sia il reparto sia il sentiment assegnati alla recensione.
    return f"{other_sentence}, ma {target_sentence[0].lower() + target_sentence[1:]}."


def _make_review(department: str, sentiment: str, rng: random.Random) -> tuple[str, str]:
    title_draw = rng.random()
    if title_draw < 0.20:
        title = rng.choice(NEUTRAL_TITLES)
    elif title_draw < 0.42:
        title = rng.choice(GENERIC_TITLES[sentiment])
    else:
        title = rng.choice(SPECIFIC_TITLES[department][sentiment])

    if rng.random() < 0.36:
        body = _ambiguous_body(department, sentiment, rng)
        closing_probability = 0.48
    else:
        body = _sentence_for(department, sentiment, rng)
        closing_probability = 0.70

    if rng.random() < closing_probability:
        body = f"{body} {rng.choice(CLOSINGS[sentiment])}"

    # Piccole varianti ortografiche comuni, applicate con moderazione.
    if department == "Reception" and rng.random() < 0.20:
        body = body.replace("check-in", rng.choice(["check in", "check-in", "checkin"]))
        body = body.replace("check-out", rng.choice(["check out", "check-out", "checkout"]))
    if department == "F&B" and rng.random() < 0.18:
        body = body.replace("ristorante", rng.choice(["ristorante", "servizio ristorante", "sala ristorante"]))

    return title, body


def generate_reviews(n_samples: int = 360, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame bilanciato tra 200 e 500 osservazioni.

    ``n_samples`` deve essere multiplo di 6 per garantire lo stesso numero di
    recensioni per ciascuna combinazione reparto-sentiment.
    """
    if not 200 <= n_samples <= 500:
        raise ValueError("n_samples deve essere compreso tra 200 e 500")
    if n_samples % (len(DEPARTMENTS) * len(SENTIMENTS)) != 0:
        raise ValueError("n_samples deve essere multiplo di 6")

    rng = random.Random(seed)
    per_cell = n_samples // 6
    records: list[dict[str, str | int]] = []
    seen: set[tuple[str, str]] = set()

    for department in DEPARTMENTS:
        for sentiment in SENTIMENTS:
            count = 0
            attempts = 0
            while count < per_cell:
                attempts += 1
                if attempts > 50_000:
                    raise RuntimeError("Impossibile generare abbastanza recensioni uniche")
                title, body = _make_review(department, sentiment, rng)
                key = (title, body)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "title": title,
                        "body": body,
                        "department": department,
                        "sentiment": sentiment,
                    }
                )
                count += 1

    rng.shuffle(records)
    for index, row in enumerate(records, start=1):
        row["id"] = f"REV-{index:04d}"

    columns = ["id", "title", "body", "department", "sentiment"]
    return pd.DataFrame(records, columns=columns)


def export_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_example_batch(rows: Iterable[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": f"BATCH-{i:03d}", "title": title, "body": body}
            for i, (title, body) in enumerate(rows, start=1)
        ]
    )


def default_example_batch() -> pd.DataFrame:
    return build_example_batch(
        [
            ("Camera da sistemare", "Nel bagno c'erano residui e gli asciugamani erano macchiati."),
            ("Arrivo senza problemi", "Check-in rapido, personale chiaro e chiave consegnata subito."),
            ("Colazione migliorabile", "Il buffet era poco vario e il caffè è arrivato freddo."),
            ("Esperienza mista", "La camera era pulita, ma al check-out ho atteso troppo per la fattura."),
            ("Cena piacevole", "Piatti ben presentati e servizio di sala molto attento."),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "reviews_synthetic.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_reviews(n_samples=args.n_samples, seed=args.seed)
    export_dataset(df, args.output)

    batch = default_example_batch()
    batch_path = args.output.parent / "batch_example.csv"
    batch.to_csv(batch_path, index=False, encoding="utf-8-sig")

    print(f"Dataset creato: {args.output} ({len(df)} righe)")
    print(df.groupby(["department", "sentiment"]).size())
    print(f"Batch di esempio creato: {batch_path}")


if __name__ == "__main__":
    main()
