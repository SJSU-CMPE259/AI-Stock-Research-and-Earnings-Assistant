"""Agent base class. Agents read named inputs from memory and write named outputs."""
from __future__ import annotations

from agent_core.memory import SharedMemory


class MissingInput(Exception):
    """A required memory input is absent."""


class Agent:
    """Base for all agents.

    Each agent: (1) checks its inputs exist, (2) does its work, (3) writes outputs to
    memory, (4) returns a one-line summary like
    'analysis_agent: read amd_revenue_quarterly -> wrote amd_notes (3 notes)'.
    """

    name: str = "agent"
    reads: list[str] = []
    writes: list[str] = []

    def run(self, ticker: str, question: str, memory: SharedMemory, llm) -> str:
        raise NotImplementedError

    def check_inputs(self, memory: SharedMemory) -> None:
        missing = [n for n in self.reads if not memory.has(n)]
        if missing:
            raise MissingInput(f"{self.name} is missing inputs: {missing}")
