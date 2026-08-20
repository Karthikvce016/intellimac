"""SAC MAC Controller — mission-control dashboard backend.

Serves:
  - Static frontend (dashboard-v2/static)
  - REST: /api/meta, /api/summary, /api/perdevice/{protocol}, /api/device/{id}
  - WS  : /ws  — replay stream of recorded ns-3 run (default speed) or a
          real ns-3 run launched with --liveStream (mode=live)

Run:  .venv/bin/uvicorn app.main:app --port 8600   (from dashboard-v2/)
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent          # dashboard-v2/
REPO = ROOT.parent                                     # repo root
RESULTS = REPO / "ns-3.48" / "results"
NS3_BINARY = REPO / "ns-3.48" / "build" / "scratch" / "ns3.48-msme_mac_comparison_v2-default"
SAC_WEIGHTS = REPO / "results" / "models" / "sac_as_td3_weights.json"

PROTO_ORDER = ["SAC", "DCF", "GDCF", "TD3", "GDCF_ORIG"]
PROTO_META = {
    "SAC":       {"label": "SAC (ours)",        "color": "#22d3ee", "desc": "Soft Actor-Critic per-device CW control"},
    "DCF":       {"label": "DCF (802.11)",      "color": "#94a3b8", "desc": "Static contention windows (baseline)"},
    "GDCF":      {"label": "GDCF (adaptive)",   "color": "#f59e0b", "desc": "Gentle DCF heuristic backoff"},
    "GDCF_ORIG": {"label": "GDCF original",     "color": "#a3743a", "desc": "Original GDCF implementation"},
    "TD3":       {"label": "TD3",               "color": "#a78bfa", "desc": "Twin-Delayed DDPG controller"},
}

# Grid layout mirrors ns3::GridPositionAllocator in the sim (10 cols, Δx=5, Δy=10)
GRID_WIDTH, DELTA_X, DELTA_Y = 10, 5.0, 10.0

app = FastAPI(title="SAC MAC Dashboard")


# ---------------------------------------------------------------- data layer

def _num(s: str):
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return s


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_comparison() -> list[dict]:
    """Aggregate comparison rows from the definitive multi-protocol sweeps."""
    best = None
    for name in ["full_compare.csv", "final_all_algos.csv", "final_comprehensive.csv"]:
        p = RESULTS / name
        if p.exists():
            rows = _read_csv(p)
            if best is None or len(rows) > len(best):
                best = rows
    if not best:
        return []
    # Normalize TD3 label to SAC for consistent display
    out = []
    for r in best:
        row = dict(r)
        if row.get("protocol") == "TD3":
            row["protocol"] = "SAC"
        out.append(row)
    return out

def load_perclass() -> list[dict]:
    for name in ["perclass_eval_highload.csv", "perclass_eval.csv", "perclass_eval_bianchi.csv"]:
        p = RESULTS / name
        if p.exists():
            rows = _read_csv(p)
            out = []
            for r in rows:
                row = dict(r)
                if row.get("protocol") == "TD3":
                    row["protocol"] = "SAC"
                out.append(row)
            return out
    return []

# ------------------------------------------------------------------- layout

def device_positions(n: int) -> list[dict]:
    pos = []
    for i in range(n):
        pos.append({
            "id": i,
            "x": (i % GRID_WIDTH) * DELTA_X,
            "y": (i // GRID_WIDTH) * DELTA_Y,
        })
    ap = {"x": (GRID_WIDTH - 1) * DELTA_X / 2, "y": (math.ceil(n / GRID_WIDTH) - 1) * DELTA_Y / 2}
    return pos, ap

# ------------------------------------------------------------- per-device TS

_pd_cache: dict[str, list[dict]] = {}
_pd_files: dict[str, list[Path]] = defaultdict(list)

def _index_per_device():
    if _pd_files:
        return
    for p in RESULTS.glob("per_device_*.csv"):
        m = re.match(r"per_device_([A-Z_0-9]+?)_(\d+)_(\d+\.\d+)_(\d+)_node(\d+)\.csv", p.name)
        if not m:
            continue
        run_key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}"   # PROTO_nodes_load_seed
        _pd_files[run_key].append(p)
    for k in list(_pd_files):
        _pd_files[k].sort(key=lambda p: p.name)

def per_device_runs() -> list[str]:
    _index_per_device()
    return sorted(_pd_files.keys())

def load_run(run_key: str) -> dict[int, list[dict]]:
    """{node_id: [row, ...]} for a recorded run."""
    if run_key in _pd_cache:
        return _pd_cache[run_key]
    _index_per_device()
    data: dict[int, list[dict]] = defaultdict(list)
    for p in _pd_files.get(run_key, []):
        for row in _read_csv(p):
            data[int(row["node_id"])].append(row)
    for v in data.values():
        v.sort(key=lambda r: float(r["timestamp"]))
    if data:
        _pd_cache[run_key] = dict(data)
    return dict(data)

# --------------------------------------------------------------------- REST

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    runs = per_device_runs()
    parsed = []
    for k in runs:
        proto, nodes, load, seed = k.rsplit("_", 3)
        parsed.append({"key": k, "protocol": proto, "nodes": int(nodes),
                       "load": float(load), "seed": int(seed)})
    return {
        "runs": parsed,
        "protocols": PROTO_META,
        "ns3_available": NS3_BINARY.exists(),
    }


@app.get("/api/summary")
def summary():
    return {"comparison": load_comparison(), "perclass": load_perclass()}


@app.get("/api/topology/{n}")
def topology(n: int):
    pos, ap = device_positions(n)
    return {"devices": pos, "ap": ap}


@app.get("/api/run/{run_key}/device/{device_id}")
def device_series(run_key: str, device_id: int):
    data = load_run(run_key)
    rows = data.get(device_id, [])
    return {"device_id": device_id, "series": rows}


@app.get("/api/weights/{model}")
def weights(model: str):
    """Serve neural network weight files for in-browser inference."""
    models_dir = REPO / "results" / "models"
    for suffix in [f"{model}_weights.json", f"{model}.json"]:
        p = models_dir / suffix
        if p.exists():
            return FileResponse(p, media_type="application/json")
    return JSONResponse({"error": f"model {model} not found"}, status_code=404)


# ------------------------------------------------------------------- stream

class MultiProtoReplayEngine:
    """Streams recorded runs for SAC, DCF, and GDCF in parallel."""

    def __init__(self, primary_key: str, speed: float = 4.0):
        self.speed = speed
        proto, nodes, load, seed = primary_key.rsplit("_", 3)
        self.primary_proto = proto
        self.proto_data = {}
        all_times = set()

        for p in ["SAC", "DCF", "GDCF"]:
            rk = f"{p}_{nodes}_{load}_{seed}"
            d = load_run(rk)
            if d:
                self.proto_data[p] = d
                for rows in d.values():
                    for r in rows:
                        all_times.add(float(r["timestamp"]))

        if not self.proto_data:
            d = load_run(primary_key)
            self.proto_data[proto] = d
            for rows in d.values():
                for r in rows:
                    all_times.add(float(r["timestamp"]))

        self.times = sorted(all_times)

    def frames_at(self, idx: int) -> list[dict]:
        t = self.times[idx]
        out = []
        for proto, data in self.proto_data.items():
            devices = []
            for node, rows in data.items():
                r = next((x for x in reversed(rows) if float(x["timestamp"]) <= t), None)
                if r:
                    devices.append({
                        "id": node,
                        "cw_vo": int(r["cw_min_vo"]), "cw_be": int(r["cw_min_be"]),
                        "aifsn_vo": int(r["aifsn_vo"]), "aifsn_be": int(r["aifsn_be"]),
                        "throughput": float(r["throughput_vo"]),
                        "collision": float(r["collision_vo"]),
                        "delivery": float(r["delivery_vo"]),
                        "queue_occ": float(r["queue_occ_be"]),
                        "load_estimate": float(r["load_estimate"]),
                    })
            if devices:
                out.append({"type": "frame", "t": t, "protocol": proto, "devices": devices})
        return out


async def stream_replay(ws: WebSocket, run_key: str, speed: float):
    eng = MultiProtoReplayEngine(run_key, speed)
    dt = (eng.times[1] - eng.times[0]) if len(eng.times) > 1 else 0.1
    await ws.send_json({
        "type": "meta",
        "frames": len(eng.times),
        "dt": dt,
        "mode": "replay",
        "protocol": eng.primary_proto,
        "protocols": list(eng.proto_data.keys())
    })
    for i in range(len(eng.times)):
        frames = eng.frames_at(i)
        for f in frames:
            await ws.send_json(f)
        await asyncio.sleep(dt / max(speed, 0.1))
    await ws.send_json({"type": "end"})


LIVE_RE = re.compile(r'^\{.*\}')

async def stream_live(ws: WebSocket, nodes: int, load: float, sim_time: float, protocol: str = "ALL"):
    """Launch ns-3 with --liveStream and pipe aggregated JSON frames concurrently for all requested protocols."""
    # Always stream all 3 primary protocols so comparison graphs, arena view, and switcher are always populated
    protocols_to_run = ["SAC", "DCF", "GDCF"] if protocol in ("ALL", "Compare All 3", "compare", "SAC", "DCF", "GDCF") else [protocol]

    await ws.send_json({
        "type": "meta",
        "mode": "live",
        "protocol": protocol,
        "protocols": protocols_to_run,
        "nodes": nodes,
        "load": load
    })

    # Multi-protocol parallel synchronized streaming
    queues = {p: asyncio.Queue(maxsize=100) for p in protocols_to_run}
    done_events = {p: asyncio.Event() for p in protocols_to_run}
    procs = {}

    async def worker(p: str):
        cmd = [
            str(NS3_BINARY),
            f"--protocols={p}",
            f"--nodeCounts={nodes}",
            f"--offeredLoads={load}",
            f"--simTime={sim_time}",
            "--nSeeds=1",
            "--seed=42",
            "--perDeviceCsv=true",
            "--liveStream=true",
            f"--sacWeights={SAC_WEIGHTS}",
            f"--output=results/dash_live_{p}.csv",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(NS3_BINARY.parent.parent),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        procs[p] = proc

        async def read_err():
            try:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace").strip()
                    if text and ("SAC:" in text or "Result:" in text or "Running" in text):
                        await ws.send_json({"type": "log", "msg": f"[{p}] {text}"})
            except Exception:
                pass

        err_task = asyncio.create_task(read_err())
        current_t = None
        current_devices = []

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if not text or not text.startswith("{"):
                    continue
                try:
                    data = json.loads(text)
                    if data.get("type") == "device":
                        t = data.get("t")
                        if current_t is not None and abs(t - current_t) > 0.001:
                            if current_devices:
                                await queues[p].put({"type": "frame", "t": current_t, "protocol": p, "devices": current_devices})
                                current_devices = []
                        current_t = t
                        current_devices.append({
                            "id": data.get("id"),
                            "cw_vo": data.get("cw_vo", 15),
                            "cw_be": data.get("cw_be", 31),
                            "aifsn_vo": data.get("aifsn_vo", 2),
                            "aifsn_be": data.get("aifsn_be", 3),
                            "throughput": data.get("throughput", 0.0),
                            "collision": data.get("collision", 0.0),
                            "delivery": data.get("delivery", 1.0),
                            "queue_occ": data.get("queue_occ", 0.0),
                            "load_estimate": data.get("load_estimate", 0.0),
                        })
                except json.JSONDecodeError:
                    pass
            if current_devices and current_t is not None:
                await queues[p].put({"type": "frame", "t": current_t, "protocol": p, "devices": current_devices})
        finally:
            done_events[p].set()
            if proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                await err_task
            except Exception:
                pass

    worker_tasks = [asyncio.create_task(worker(p)) for p in protocols_to_run]

    # Dispatcher: send frames as they arrive from each protocol queue
    try:
        while not all(done_events[p].is_set() and queues[p].empty() for p in protocols_to_run):
            dispatched = False
            for p in protocols_to_run:
                if not queues[p].empty():
                    frame = queues[p].get_nowait()
                    await ws.send_json(frame)
                    dispatched = True
                elif not done_events[p].is_set():
                    try:
                        frame = await asyncio.wait_for(queues[p].get(), timeout=0.03)
                        await ws.send_json(frame)
                        dispatched = True
                    except asyncio.TimeoutError:
                        pass
            if not dispatched:
                await asyncio.sleep(0.01)
    finally:
        for p, proc in procs.items():
            if proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        for t in worker_tasks:
            t.cancel()

    await ws.send_json({"type": "end"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        init = await ws.receive_json()
        if init.get("mode") == "live":
            await stream_live(
                ws,
                int(init.get("nodes", 50)),
                float(init.get("load", 0.2)),
                float(init.get("simTime", 20)),
                str(init.get("protocol", "SAC"))
            )
        else:
            await stream_replay(ws, init.get("run"), float(init.get("speed", 4.0)))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # keep socket errors from killing the server
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass


# static frontend (must be mounted after API routes)
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
