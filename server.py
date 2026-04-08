"""
server.py — OpenEnv HTTP server for Lead Qualification environment
Exposes: POST /reset, POST /step, GET /state, GET /health
"""
from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import LeadQualificationEnv

app = FastAPI(title="Prospect Intelligence — OpenEnv", version="1.0.0")
_env: Optional[LeadQualificationEnv] = None

def _get_env() -> LeadQualificationEnv:
    global _env
    if _env is None:
        _env = LeadQualificationEnv()
    return _env

class StepRequest(BaseModel):
    action: Dict[str, Any]

def _serialise(obj: Any) -> Any:
    if isinstance(obj, dict):   return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_serialise(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None: return obj
    return str(obj)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/reset")
def reset(body: Dict = {}):
    global _env
    _env = LeadQualificationEnv()
    return JSONResponse(content=_serialise(_env.reset()))

@app.post("/step")
def step(req: StepRequest):
    env = _get_env()
    try:
        obs, reward, done, info = env.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content={
        "observation": _serialise(obs),
        "reward": reward,
        "done": done,
        "info": _serialise(info),
    })

@app.get("/state")
def state():
    return JSONResponse(content=_serialise(_get_env().state()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)