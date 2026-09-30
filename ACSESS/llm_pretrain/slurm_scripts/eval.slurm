#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -A cis260281p
#SBATCH --mem=48GB
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH -J llm_eval

set -e

# Load environment
module load anaconda3
conda activate mlenv

# Paths
PROJECT_DIR="/ocean/projects/cis260281p/llm_pretrain"
SRC_DIR="${PROJECT_DIR}/src"
CKPT_DIR="${PROJECT_DIR}/checkpoints"
EVAL_DIR="${PROJECT_DIR}/eval_results"

mkdir -p ${EVAL_DIR} logs

# Find latest checkpoint
CKPT=$(ls -t ${CKPT_DIR}/ckpt_*.pt 2>/dev/null | head -1)

if [ -z "${CKPT}" ]; then
    echo "No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

TOKENIZER="${PROJECT_DIR}/data/tokenizer.json"

if [ ! -f "${TOKENIZER}" ]; then
    echo "Tokenizer not found at ${TOKENIZER}"
    exit 1
fi

echo "Evaluating checkpoint: ${CKPT}"
echo "Using tokenizer: ${TOKENIZER}"

# Export model
EXPORT_DIR="${EVAL_DIR}/exported_model"
python ${SRC_DIR}/eval.py \
  --checkpoint ${CKPT} \
  --tokenizer ${TOKENIZER} \
  --output_dir ${EXPORT_DIR} \
  --verify

# Run lm-eval tasks
echo "Running evaluation tasks..."
python -c "
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

tasks = [
    'arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'social_iqa',
    'winogrande', 'commonsense_qa', 'openbookqa', 'boolq', 'lambada_openai'
]

model = HFLM(pretrained='${EXPORT_DIR}', dtype='float16', device='cuda')
results = evaluator.evaluate(model, tasks, num_fewshot=0, batch_size=8, device='cuda')

import json
with open('${EVAL_DIR}/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"

echo "Evaluation complete! Results saved to ${EVAL_DIR}/eval_results.json"
