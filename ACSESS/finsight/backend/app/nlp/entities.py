"""spaCy NER with explicit company alias and ticker normalization."""

from collections.abc import Iterable

import spacy
from spacy.language import Language

from .schemas import Entity, Passage, Statement, stable_id

COMPANIES = {
    "NVDA": ("NVIDIA", "NVIDIA Corporation"),
    "AAPL": ("Apple", "Apple Inc."),
    "MSFT": ("Microsoft", "Microsoft Corporation"),
    "AMZN": ("Amazon", "Amazon.com"),
    "GOOGL": ("Alphabet", "Alphabet Inc.", "Google"),
    "META": ("Meta", "Meta Platforms"),
    "TSLA": ("Tesla", "Tesla Inc."),
    "JPM": ("JPMorgan Chase", "JPMorgan"),
    "AMD": ("Advanced Micro Devices",),
    "INTC": ("Intel", "Intel Corporation"),
}


def build_pipeline(model_name: str | None = "en_core_web_sm") -> Language:
    """None selects an explicit rules-only baseline; never silently downgrade."""
    nlp = spacy.load(model_name) if model_name else spacy.blank("en")
    if not any(name in nlp.pipe_names for name in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    ruler = (
        nlp.add_pipe("entity_ruler", before="ner")
        if "ner" in nlp.pipe_names
        else nlp.add_pipe("entity_ruler")
    )
    patterns = []
    for ticker, aliases in COMPANIES.items():
        patterns.append(
            {"label": "TICKER", "pattern": [{"TEXT": ticker}], "id": ticker}
        )
        patterns.extend(
            {
                "label": "ORG",
                "pattern": [
                    {"LOWER": token.text.lower()} for token in nlp.make_doc(alias)
                ],
                "id": ticker,
            }
            for alias in aliases
        )
    ruler.add_patterns(patterns)
    return nlp


def extract_statements(
    passages: Iterable[Passage], nlp: Language
) -> tuple[Statement, ...]:
    """Keep offsets relative to the supplied passage, not the original filing HTML."""
    passages = tuple(passages)
    statements = []
    for passage, doc in zip(passages, nlp.pipe(p.text for p in passages)):
        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            start = (
                sentence.start_char + len(sentence.text) - len(sentence.text.lstrip())
            )
            end = sentence.end_char - (len(sentence.text) - len(sentence.text.rstrip()))
            entities = tuple(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    canonical_ticker=ent.ent_id_ or None,
                )
                for ent in sentence.ents
                if start <= ent.start_char and ent.end_char <= end
            )
            statements.append(
                Statement(
                    statement_id=stable_id(
                        "S", passage.passage_id, str(start), str(end)
                    ),
                    passage_id=passage.passage_id,
                    ticker=passage.ticker,
                    text=passage.text[start:end],
                    start=start,
                    end=end,
                    entities=entities,
                )
            )
    return tuple(statements)
