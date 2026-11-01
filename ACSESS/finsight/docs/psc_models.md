# Running the models on PSC (Mistral :8000 + Laya :8001)

FinSight's live mode uses two open models on a PSC Bridges-2 GPU, reached from your
laptop through SSH tunnels. If either is down, the app degrades gracefully (keyword
routing instead of Laya; sample quote instead of Finnhub is separate — see .env).

**Allocation:** the GPU job runs on the `cis260281p` grant (owner: mrajpurohit). To run
it yourself you need to be added to that PSC grant by the PI — ask Meenakshi. Otherwise
coordinate a session where she brings the servers up and you tunnel in.

Everything below runs **on the GPU node** (`vNNN`/`wNNN`), never a login node. `/tmp` is
per-node and wiped between jobs, so this is a fresh setup each session.

## 1. Get a GPU node
```bash
ssh <user>@bridges2.psc.edu
salloc -p GPU-shared -N 1 --gres=gpu:1 -t 02:00:00   # prompt becomes [user@NODE ~]$
```
Note the node name `NODE` (e.g. `w008`). `nvidia-smi` should show a V100 or H100.

## 2. Mistral on :8000 (ollama, OpenAI-compatible)
```bash
cd /tmp && curl -L https://github.com/ollama/ollama/releases/download/v0.40.3/ollama-linux-amd64.tar.zst -o ollama.tar.zst \
  && zstd -d ollama.tar.zst -o ollama.tar && mkdir -p ollama-bin && tar -xf ollama.tar -C ollama-bin
export OLLAMA_MODELS=/tmp/ollama OLLAMA_HOST=0.0.0.0:8000 LD_LIBRARY_PATH=/tmp/ollama-bin/lib/ollama
/tmp/ollama-bin/bin/ollama serve          # leave running
# in a second shell on the SAME node (srun --jobid=$(squeue -u <user> -h -o %i|head -1) --pty bash):
OLLAMA_HOST=0.0.0.0:8000 /tmp/ollama-bin/bin/ollama pull mistral
```

## 3. Laya on :8001
In that second node shell, set up the env and run the server (see `scripts/serve_laya.py`):
```bash
export TMPDIR=/tmp HF_HOME=/tmp/hf PIP_CACHE_DIR=/tmp/pip CONDA_PKGS_DIRS=/tmp/condapkgs
module load anaconda3
conda create -p /tmp/laya-env python=3.10 -y && conda activate /tmp/laya-env
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "transformers<5" accelerate laya fastapi uvicorn
python scripts/serve_laya.py              # -> Uvicorn running on http://0.0.0.0:8001
```

## 4. Tunnels (from your laptop; replace NODE)
```bash
ssh -N -L 8000:NODE.ib.bridges2.psc.edu:8000 <user>@bridges2.psc.edu   # Mistral
ssh -N -L 8001:NODE.ib.bridges2.psc.edu:8001 <user>@bridges2.psc.edu   # Laya
```
These hang silently — that's correct. If a tunnel errors with "Name or service not
known", use the node IP instead: run `hostname -i` on the node and tunnel to that IP.

## 5. Verify + point the app at them
```bash
curl http://localhost:8000/v1/models    # -> lists "mistral"
curl http://localhost:8001/health        # -> {"ok": true}
```
`.env` (Mistral is the `remote` provider; Laya is just the tunnel endpoint):
```
LLM_PROVIDER=remote
REMOTE_LLM_URL=http://localhost:8000/v1
LLM_MODEL=mistral
LAYA_URL=http://localhost:8001          # default; only needed if you change the port
```
