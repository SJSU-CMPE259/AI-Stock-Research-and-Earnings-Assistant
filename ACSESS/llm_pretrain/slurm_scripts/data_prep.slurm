#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=0
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -A cis260281p
#SBATCH --mem=64GB
#SBATCH -o logs/data_prep_%j.out
#SBATCH -e logs/data_prep_%j.err
#SBATCH -J data_prep
#SBATCH --dependency=afterok:TOKENIZER_JOB_ID

set -e

# Load environment
module load anaconda3
conda activate mlenv

# Paths
PROJECT_DIR="/ocean/projects/cis260281p/llm_pretrain"
DATA_DIR="${PROJECT_DIR}/data"
SRC_DIR="${PROJECT_DIR}/src"

mkdir -p ${DATA_DIR} logs

# Prepare data
echo "Starting data preparation..."
python ${SRC_DIR}/data_prep.py \
  --tokenizer_path ${DATA_DIR}/tokenizer.json \
  --output_dir ${DATA_DIR}/shards \
  --total_tokens 25000000000 \
  --vocab_size 50304

echo "Data preparation complete!"
