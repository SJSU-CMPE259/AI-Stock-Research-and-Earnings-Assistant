"""Serve the Laya router over HTTP on :8001 (run this ON the PSC GPU node).

Laya makes the routing decision (intent + advice); it is a typed-decision model, not a
generator. The orchestrator's LayaRouter POSTs {state, questions} to /predict and reads
back the chosen route + confidence. If this server is down, the orchestrator falls back
to keyword routing automatically.

Setup on the GPU node (inside your salloc/srun shell), once per node:
    export TMPDIR=/tmp HF_HOME=/tmp/hf PIP_CACHE_DIR=/tmp/pip CONDA_PKGS_DIRS=/tmp/condapkgs
    module load anaconda3
    conda create -p /tmp/laya-env python=3.10 -y && conda activate /tmp/laya-env
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install "transformers<5" accelerate laya fastapi uvicorn
    python scripts/serve_laya.py          # -> Uvicorn running on http://0.0.0.0:8001

Then, from your laptop, tunnel it (replace NODE with your node, e.g. w008 or v001):
    ssh -N -L 8001:NODE.ib.bridges2.psc.edu:8001 <user>@bridges2.psc.edu
The app reads LAYA_URL (default http://localhost:8001); keep the default once tunnelled.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from laya import Router

app = FastAPI()
router = Router()


class Req(BaseModel):
    state: str          # the question
    questions: dict     # typed decisions (e.g. advice=noul, intent=choice+criteria)


@app.post("/predict")
def predict(r: Req):
    return router.predict(r.state, r.questions)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
