import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from fraud.data import load, time_split
from fraud.features import RAW_COLUMNS
from fraud.serve import load_bundle, score_record

N_REQUESTS = 2000
WARMUP = 200
PORT = 8765


def percentiles(ms):
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "n": len(ms)}


def timed(fn, records):
    out = []
    for r in records:
        t = time.perf_counter()
        fn(r)
        out.append((time.perf_counter() - t) * 1000)
    return out


def wait_for_server(client):
    for _ in range(100):
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def main():
    _, _, test = time_split(load())
    sample = test.sample(N_REQUESTS + WARMUP, random_state=0)
    records = sample[RAW_COLUMNS].to_dict("records")

    bundle = load_bundle()
    in_process = timed(lambda r: score_record(bundle, r), records[WARMUP:])

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "fraud.serve:app", "--port", str(PORT),
                               "--log-level", "warning"])
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}") as client:
            wait_for_server(client)
            for r in records[:WARMUP]:
                client.post("/score", json=r)
            http = timed(lambda r: client.post("/score", json=r).raise_for_status(), records[WARMUP:])
    finally:
        server.terminate()
        server.wait()

    result = {"model": bundle["name"], "in_process_scoring": percentiles(in_process),
              "http_end_to_end": percentiles(http)}
    Path("reports/latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
