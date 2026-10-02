"""Utility comuni per la normalizzazione del testo.

Il modulo e' volutamente semplice: il project work richiede una pipeline
leggibile e riproducibile, senza dipendenze NLP pesanti.
"""
from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\sàèéìòù]", flags=re.IGNORECASE)


def normalize_text(text: object) -> str:
    """Normalizza un testo per la vettorizzazione.

    Operazioni applicate:
    1. conversione in stringa;
    2. normalizzazione Unicode NFKC;
    3. lowercasing;
    4. sostituzione della punteggiatura con spazi;
    5. compressione degli spazi multipli.

    Gli accenti italiani vengono mantenuti. La funzione non rimuove le
    stopword: in un dataset piccolo parole come "non" sono informative per
    il sentiment e non devono essere eliminate automaticamente.
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = value.replace("&", " e ")
    value = _NON_WORD_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def combine_title_body(title: object, body: object) -> str:
    """Unisce titolo e corpo, mantenendo un marcatore testuale leggibile."""
    title_text = "" if title is None else str(title).strip()
    body_text = "" if body is None else str(body).strip()
    return f"titolo {title_text} testo {body_text}".strip()
