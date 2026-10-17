"""SharedMemory (CAVM) and Conversation for the FinSight agent core.

Agents never call each other or pass data directly. They read named items from
SharedMemory and write their results back under new names. describe() returns a
one-line index so the LLM sees a small summary instead of all the data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    value: Any
    written_by: str
    sources: list[dict] = field(default_factory=list)
    freshness: str = "stored"  # "stored" | "real_time"
    description: str = ""
    created_at: float = field(default_factory=time.time)


class SharedMemory:
    """One shared variable space for a run; named items with provenance."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def put(
        self,
        name: str,
        value: Any,
        written_by: str,
        sources: list[dict] | None = None,
        freshness: str = "stored",
        description: str = "",
    ) -> None:
        self._items[name] = MemoryItem(
            value=value,
            written_by=written_by,
            sources=sources or [],
            freshness=freshness,
            description=description,
        )

    def get(self, name: str) -> Any:
        item = self._items.get(name)
        return item.value if item else None

    def item(self, name: str) -> MemoryItem | None:
        return self._items.get(name)

    def has(self, name: str) -> bool:
        return name in self._items

    def list(self) -> list[str]:
        return list(self._items.keys())

    def describe(self) -> str:
        """One short line per item: name, who wrote it, freshness, short description."""
        lines = []
        for name, it in self._items.items():
            desc = it.description or _short(it.value)
            lines.append(f"- {name} (by {it.written_by}, {it.freshness}): {desc}")
        return "\n".join(lines)

    def age_seconds(self, name: str) -> float | None:
        it = self._items.get(name)
        return None if it is None else time.time() - it.created_at

    def save(self, path: str | Path) -> None:
        """Dump the whole memory to JSON so a run can be inspected afterwards."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {
                "value": it.value,
                "written_by": it.written_by,
                "sources": it.sources,
                "freshness": it.freshness,
                "description": it.description,
                "created_at": it.created_at,
            }
            for name, it in self._items.items()
        }
        path.write_text(json.dumps(data, indent=2, default=str))


def _short(value: Any, limit: int = 70) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


@dataclass
class Conversation:
    """Small per-conversation state: company, fiscal period, and last N Q/A pairs.

    Only this small state goes into prompts, never the full chat history (fixes the
    prototype bug where the whole conversation was re-sent every turn).
    """

    current_ticker: str | None = None
    current_period: str | None = None
    max_pairs: int = 3
    pairs: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.pairs.append((question, answer))
        if len(self.pairs) > self.max_pairs:
            self.pairs = self.pairs[-self.max_pairs :]

    def state_text(self) -> str:
        lines = []
        if self.current_ticker:
            lines.append(f"Current company: {self.current_ticker}")
        if self.current_period:
            lines.append(f"Current fiscal period: {self.current_period}")
        for q, a in self.pairs:
            lines.append(f"Q: {q}\nA: {a}")
        return "\n".join(lines)
