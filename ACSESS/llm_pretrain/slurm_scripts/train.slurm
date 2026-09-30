#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -A cis260281p
#SBATCH --mem=48GB
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -J llm_train
#SBATCH --dependency=afterok:DATA_PREP_JOB_ID

set -e

# Load environment
module load anaconda3
conda activate mlenv

# Paths
PROJECT_DIR="/ocean/projects/cis260281p/llm_pretrain"
DATA_DIR="${PROJECT_DIR}/data/shards"
SRC_DIR="${PROJECT_DIR}/src"
CKPT_DIR="${PROJECT_DIR}/checkpoints"

mkdir -p ${CKPT_DIR} logs

# Training parameters
BATCH_SIZE=256
SEQ_LEN=2048
GRADIENT_ACCUM=1
PEAK_LR=5e-4
MIN_LR=5e-5
WARMUP_STEPS=2000
TOTAL_STEPS=47700
LOG_EVERY=10
VAL_EVERY=1000
CHECKPOINT_EVERY=60

# Find latest checkpoint for resuming
RESUME_CKPT=""
if [ -f "${CKPT_DIR}/latest_checkpoint.txt" ]; then
    RESUME_CKPT=$(cat ${CKPT_DIR}/latest_checkpoint.txt)
    if [ ! -f "${RESUME_CKPT}" ]; then
        RESUME_CKPT=""
    fi
fi

echo "Starting training..."
echo "Data dir: ${DATA_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
if [ -n "${RESUME_CKPT}" ]; then
    echo "Resuming from: ${RESUME_CKPT}"
fi

# Run training
python -m torch.distributed.launch --nproc_per_node=1 ${SRC_DIR}/train.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${CKPT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --seq_len ${SEQ_LEN} \
  --gradient_accumulation_steps ${GRADIENT_ACCUM} \
  --peak_lr ${PEAK_LR} \
  --min_lr ${MIN_LR} \
  --warmup_steps ${WARMUP_STEPS} \
  --total_steps ${TOTAL_STEPS} \
  --log_every ${LOG_EVERY} \
  --val_every ${VAL_EVERY} \
  --checkpoint_every ${CHECKPOINT_EVERY} \
  --max_checkpoints 3 \
  $([ -n "${RESUME_CKPT}" ] && echo "--resume_from ${RESUME_CKPT}")

# Save latest checkpoint for potential resumption
LATEST=$(ls -t ${CKPT_DIR}/ckpt_*.pt 2>/dev/null | head -1)
if [ -n "${LATEST}" ]; then
    echo "${LATEST}" > ${CKPT_DIR}/latest_checkpoint.txt
fi

echo "Training job complete!"
