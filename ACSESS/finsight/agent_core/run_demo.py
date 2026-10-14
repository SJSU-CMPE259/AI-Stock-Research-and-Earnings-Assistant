"""Run the 5-question demo and save a trace + memory snapshot.

    python -m agent_core.run_demo --fake   # offline, FakeLLM
    python -m agent_core.run_demo          # Mistral on PSC (needs the tunnel up)

Run from ACSESS/finsight/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent_core.llm import FakeLLM, MistralLLM
from agent_core.memory import Conversation, SharedMemory
from agent_core.orchestrator import Orchestrator

QUESTIONS = [
    "What was AMD's revenue last quarter?",
    "How did it change from the year before?",
    "What is the tone of AMD's recent filings?",
    "What is AMD's stock price right now?",
    "Should I buy AMD?",
]

RUNS_DIR = Path(__file__).parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight agent-core demo")
    parser.add_argument("--fake", action="store_true", help="use FakeLLM (offline, no GPU)")
    args = parser.parse_args()

    llm = FakeLLM() if args.fake else MistralLLM()
    memory = SharedMemory()
    orch = Orchestrator(llm, memory=memory, conversation=Conversation())

    header = f"# FinSight agent-core demo ({'FakeLLM' if args.fake else 'Mistral on PSC'})\n\n"
    trace_md = [header]
    last_run_id = None

    for i, q in enumerate(QUESTIONS, 1):
        tr = orch.ask(q)
        last_run_id = tr.run_id
        print(f"\n{'=' * 72}")
        print(f"Q{i}: {q}")
        print(f"   ticker={tr.ticker}  route={tr.route}  {tr.elapsed_s * 1000:.0f}ms")
        for s in tr.steps:
            print(f"     - {s}")
        if tr.reused:
            print(f"     reused from memory: {tr.reused}")
        print(f"   ANSWER: {tr.answer}")

        trace_md.append(f"## Q{i}: {q}\n\n")
        trace_md.append(f"- ticker: `{tr.ticker}`  route: `{tr.route}`  time: {tr.elapsed_s * 1000:.0f}ms\n")
        for s in tr.steps:
            trace_md.append(f"  - {s}\n")
        if tr.reused:
            trace_md.append(f"  - reused from memory: {tr.reused}\n")
        trace_md.append(f"\n**Answer:** {tr.answer}\n\n")

    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"{last_run_id}_trace.md").write_text("".join(trace_md))
    memory.save(RUNS_DIR / f"{last_run_id}_memory.json")
    print(f"\nSaved runs/{last_run_id}_trace.md and runs/{last_run_id}_memory.json")


if __name__ == "__main__":
    main()
