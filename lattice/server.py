"""Research terminal backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lattice.sim import SimConfig, Simulator, compare_strategies

WEB = Path(__file__).parent / "web"

app = FastAPI(title="Lattice", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    seed: int = 7
    horizon_s: float = Field(45.0, ge=5.0, le=180.0)
    strategy: str = "avellaneda_stoikov"
    latency_ms: float = Field(8.0, ge=0.0, le=80.0)
    noise_lambda: float = Field(8.0, ge=0.5, le=30.0)
    informed_lambda: float = Field(2.5, ge=0.0, le=20.0)
    as_gamma: float = Field(0.05, ge=0.005, le=0.4)
    naive_half_ticks: int = Field(1, ge=1, le=8)
    options_enabled: bool = False
    hedge_mode: str = "mm"
    option_iv: float = 0.22
    option_qty_short: int = 20
    snapshot_s: float = 0.12


class CompareRequest(BaseModel):
    seeds: int = Field(5, ge=2, le=12)
    horizon_s: float = Field(25.0, ge=8.0, le=60.0)
    seed: int = 11


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.post("/api/simulate")
def api_simulate(req: RunRequest) -> dict:
    two_state = req.strategy in ("glosten_milgrom", "gm", "glosten")
    cfg = SimConfig(
        seed=req.seed,
        horizon_s=req.horizon_s,
        strategy=req.strategy,
        latency_ms=req.latency_ms,
        noise_lambda=req.noise_lambda,
        informed_lambda=req.informed_lambda,
        as_gamma=req.as_gamma,
        naive_half_ticks=req.naive_half_ticks,
        options_enabled=req.options_enabled,
        hedge_mode=req.hedge_mode,
        option_iv=req.option_iv,
        option_qty_short=req.option_qty_short,
        snapshot_s=req.snapshot_s,
        value_process="two_state" if two_state else "bm",
        max_frames=420,
    )
    return Simulator(cfg).run().to_jsonable()


@app.post("/api/compare")
def api_compare(req: CompareRequest) -> dict:
    rows = compare_strategies(
        SimConfig(horizon_s=req.horizon_s, snapshot_s=0.25, max_frames=40),
        n_seeds=req.seeds,
        seed0=req.seed,
    )
    return {"rows": rows}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
