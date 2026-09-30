# LLM Pretraining on Bridges-2

A complete implementation of 250M-parameter Llama-style decoder-only language model pretraining on NSF ACCESS HPC (Bridges-2).

## Model Specs

- **Architecture**: Llama-style decoder with RoPE, pre-norm RMSNorm, SwiGLU MLP, no biases, tied embeddings
- **Size**: 16 layers, d_model=1024, 16 heads (64 dim each), SwiGLU hidden=2816, context=2048, vocab=50,304
- **Parameters**: ~257M
- **Training data**: 25B tokens from mixed sources (FineWeb-Edu, DCLM, German web text, code, math)
- **Tokenizer**: Byte-level BPE, vocab=50,304 (trained on dataset sample)

## Hardware

- **GPU**: Tesla V100-SXM2-32GB (Volta, fp16 with Tensor Cores)
- **Allocation**: 99 GPU SU (~99 GPU-hours), 100 GB storage on `/ocean/projects/cis260281p/`
- **Cluster**: Bridges-2 (PSC), Slurm scheduler

## Quick Start

### 1. Copy project to Bridges-2

```bash
scp -r llm_pretrain/ mrajpurohit@bridges2.psc.edu:/ocean/projects/cis260281p/
```

### 2. SSH to Bridges-2

```bash
ssh mrajpurohit@bridges2.psc.edu
cd /ocean/projects/cis260281p/llm_pretrain
```

### 3. Set up environment (one-time)

```bash
module load anaconda3
conda create -n mlenv python=3.10 -y
conda activate mlenv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets tokenizers transformers tensorboard tqdm wandb lm-eval
```

### 4. Train tokenizer (CPU job, ~30 min)

```bash
sbatch slurm_scripts/tokenizer.slurm
```

Wait for it to complete, then check:
```bash
ls -lh data/tokenizer.json
```

### 5. Prepare data (CPU job, ~6-10 hours)

```bash
TOKENIZER_JID=$(sbatch slurm_scripts/tokenizer.slurm | awk '{print $4}')
sed -i "s/TOKENIZER_JOB_ID/${TOKENIZER_JID}/" slurm_scripts/data_prep.slurm
sbatch slurm_scripts/data_prep.slurm
```

Wait for data prep to complete, then verify:
```bash
ls -lh data/shards/shard_*.pt | head -5
```

### 6. Smoke test: tiny model (15 min, ~0.2 GPU SU)

Edit `slurm_scripts/train.slurm` to reduce model size for a quick test:
```bash
# In train.slurm, add this before python command:
export TINY_MODEL=1

# In src/train.py, add at top of main():
if os.getenv("TINY_MODEL"):
    args.total_steps = 100  # Just 100 steps
```

Then:
```bash
sbatch slurm_scripts/train.slurm
tail -f logs/train_*.out
```

Kill it after 15 min (`scancel <jobid>`) and verify:
1. Loss is decreasing
2. Checkpoint was saved: `ls checkpoints/ckpt_*.pt`
3. Resume works: resubmit same job, confirm it loads checkpoint

### 7. Estimate full training cost

After smoke test, check:
```bash
cat checkpoints/metrics.csv
```

Calculate: tokens/sec × total_tokens (25B) ÷ 3600 = GPU hours needed.

**Expected**: ~20-30 GPU hours on V100 (accounting for fp16 speedup).

### 8. Launch full training

Chain jobs to run continuously:
```bash
DATA_JID=$(sbatch slurm_scripts/data_prep.slurm | awk '{print $4}')

TRAIN_JID=$(sbatch --dependency=afterok:${DATA_JID} slurm_scripts/train.slurm | awk '{print $4}')

# Submit 2-3 chained training jobs in case first one doesn't finish all steps
sbatch --dependency=afterany:${TRAIN_JID} slurm_scripts/train.slurm
sbatch --dependency=afterany:${TRAIN_JID} slurm_scripts/train.slurm
```

Monitor progress:
```bash
watch -n 60 'tail -20 logs/train_*.out'
squeue -u mrajpurohit
cat checkpoints/metrics.csv | tail -20
```

### 9. Evaluate

Once training completes (or at 5B-token milestones):
```bash
sbatch slurm_scripts/eval.slurm
```

Results saved to `eval_results/eval_results.json`.

## File Structure

```
llm_pretrain/
├── src/
│   ├── model.py              # Llama model definition
│   ├── train.py              # Training loop with DDP, fp16, checkpointing
│   ├── tokenizer_train.py    # BPE tokenizer training
│   ├── data_prep.py          # Dataset download & tokenization
│   ├── eval.py               # Model export & lm-eval harness
│   └── finetune.py           # Fine-tuning skeleton
├── slurm_scripts/
│   ├── tokenizer.slurm       # Tokenizer training job
│   ├── data_prep.slurm       # Data preparation job
│   ├── train.slurm           # Training job (chainable)
│   └── eval.slurm            # Evaluation job
├── data/
│   ├── tokenizer.json        # Trained tokenizer (after step 4)
│   └── shards/               # Tokenized dataset shards (after step 5)
├── checkpoints/              # Model checkpoints & optimizer state
├── eval_results/             # Evaluation results & exported model
└── README.md
```

## Training Details

### Hyperparameters

- **Optimizer**: AdamW (β1=0.9, β2=0.95, eps=1e-8, weight_decay=0.1)
- **Learning rate**: Warmup 2K steps to 5e-4, cosine decay to 5e-5 over 47.7K steps
- **Batch size**: 256 sequences × 2048 tokens = 512K tokens/step
- **Gradient accumulation**: 1 (full batch fits in 32GB V100)
- **Gradient clipping**: 1.0
- **fp16**: Yes, with GradScaler; final softmax/CE in fp32

### Checkpointing

- Every 60 wall-clock minutes
- Keeps last 3 checkpoints + milestone (every 5B tokens)
- Auto-resume on resubmit: finds latest checkpoint automatically

### Logging

- **CSV**: `checkpoints/metrics.csv` (loss, LR, tokens/sec, MFU, grad scale)
- **TensorBoard**: `checkpoints/logs/` (run `tensorboard --logdir=checkpoints/logs`)
- **Console**: every 10 steps

### MFU Calculation

Model FLOPs Utilization (V100 fp16 Tensor Core peak = 125 TFLOPS):
```
MFU = (tokens_seen * vocab_size * 2) / (wall_time_seconds * 125e12)
```

Expected MFU: 30-50% on single V100 with this batch size.

## Fine-tuning (Later)

```bash
python src/finetune.py \
  --checkpoint checkpoints/ckpt_final.pt \
  --train_data data/tasks/sst2.txt \
  --tokenizer data/tokenizer.json \
  --output_dir checkpoints/sst2_finetuned \
  --num_epochs 3
```

Supports GLUE (SST-2), SQuAD, CNN/DailyMail, XSum, WMT14 English-German, etc.

## Troubleshooting

**torch.compile fails**: Falls back to eager mode (slower but works). Remove compile calls if needed.

**GradScaler scale keeps dropping**: Lower peak_lr to 3e-4 in train.slurm.

**OOM on single GPU**: Reduce batch_size or gradient_accumulation_steps in train.slurm.

**Compute nodes no internet**: Download datasets on login node instead; data_prep will use cache.

**Job times out**: Wall-clock time limits are handled by dependency chains—just resubmit.

## References

- Model: Llama architecture (arXiv:2302.13971)
- Training: Chinchilla compute-optimal scaling
- Evaluation: EleutherAI lm-evaluation-harness
- Data: FineWeb, DCLM, StarCoder, FineMath

## Contact

For issues on Bridges-2: help@psc.edu
