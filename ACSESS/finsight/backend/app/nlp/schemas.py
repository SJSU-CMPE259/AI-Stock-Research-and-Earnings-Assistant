"""Source-preserving input and output contracts for analysis."""

from datetime import date
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


def stable_id(prefix: str, *parts: str) -> str:
    """Length-prefix components so different boundaries cannot collide."""
    content = "".join(f"{len(part)}:{part}" for part in parts)
    return prefix + sha256(content.encode()).hexdigest()[:16]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Passage(Record):
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]*$")
    source_type: Literal["filing", "news", "earnings_call", "synthetic"]
    source_url: HttpUrl | None = None
    accession: str | None = Field(default=None, pattern=r"^\d{10}-\d{2}-\d{6}$")
    form: str | None = None
    period_end: date | None = None
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self):
        if not self.text.strip():
            raise ValueError("Passage text must contain non-whitespace characters")
        if self.source_type != "synthetic" and self.source_url is None:
            raise ValueError("Real evidence must have a source URL")
        if self.source_type == "filing" and not all(
            (self.accession, self.form, self.period_end)
        ):
            raise ValueError("Filings require accession, form, and period_end")
        return self

    @property
    def passage_id(self) -> str:
        return stable_id("P", self.model_dump_json())


class Entity(Record):
    text: str
    label: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    canonical_ticker: str | None = None


class Statement(Record):
    statement_id: str
    passage_id: str
    ticker: str
    text: str
    start: int
    end: int
    entities: tuple[Entity, ...]


class EvidenceNote(Record):
    note_id: str
    text: str
    source_ids: tuple[str, ...]
    kind: Literal["evidence_excerpt"] = "evidence_excerpt"


class AnalysisResult(Record):
    pipeline: str
    sources: dict[str, Passage]
    statements: tuple[Statement, ...]
    notes: tuple[EvidenceNote, ...]

    @model_validator(mode="after")
    def check_evidence(self):
        ids = set()
        for statement in self.statements:
            source = self.sources.get(statement.passage_id)
            if source is None or statement.statement_id in ids:
                raise ValueError("Statements require a unique ID and a known passage")
            ids.add(statement.statement_id)
            if not 0 <= statement.start < statement.end <= len(source.text):
                raise ValueError("Statement offsets are outside the passage")
            if source.text[statement.start : statement.end] != statement.text:
                raise ValueError("Statement text must match its evidence span")
            for entity in statement.entities:
                if not statement.start <= entity.start < entity.end <= statement.end:
                    raise ValueError("Entity span is outside the statement")
                if source.text[entity.start : entity.end] != entity.text:
                    raise ValueError("Entity text must match its evidence span")
        if any(
            not note.source_ids or not set(note.source_ids) <= ids
            for note in self.notes
        ):
            raise ValueError("Every note must cite existing statement IDs")
        return self
