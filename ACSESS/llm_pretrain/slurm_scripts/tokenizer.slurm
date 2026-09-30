#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -A cis260281p
#SBATCH --mem=32GB
#SBATCH -o logs/tokenizer_%j.out
#SBATCH -e logs/tokenizer_%j.err
#SBATCH -J tokenizer

set -e

# Load environment
module load anaconda3
conda activate mlenv

# Paths
PROJECT_DIR="/ocean/projects/cis260281p/llm_pretrain"
DATA_DIR="${PROJECT_DIR}/data"
SRC_DIR="${PROJECT_DIR}/src"

mkdir -p ${DATA_DIR} logs

# Train tokenizer
echo "Starting tokenizer training..."
python ${SRC_DIR}/tokenizer_train.py \
  --output_path ${DATA_DIR}/tokenizer.json \
  --vocab_size 50304

echo "Tokenizer training complete!"
