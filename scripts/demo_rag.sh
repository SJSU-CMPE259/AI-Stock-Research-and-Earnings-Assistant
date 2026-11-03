#!/usr/bin/env bash
# Live demo: one question, three search modes. Press Enter to move to the next mode.
cd "$(dirname "$0")/.." && source .venv/bin/activate
Q="How do export limits affect Nvidia?"
for MODE in sparse dense hybrid; do
  clear
  echo "================  $MODE  ================"
  python scripts/build_rag_index.py --ask "$Q" --ticker NVDA --search --mode "$MODE" -k 5
  read -r -p "Enter for next mode..."
done