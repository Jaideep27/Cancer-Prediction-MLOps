"""Tiny load generator: N threads hammer POST /predict for D seconds.

    python -m scripts.load_test --url http://localhost:8000 --concurrency 20 --duration 60

Prints throughput, latency percentiles, and HOW MANY requests each replica
served (from the X-Served-By header) -- that is load balancing, made visible.

--new-connections opens a fresh TCP connection per request. Kubernetes Services
balance per CONNECTION, so a client that keeps one connection alive (the default)
sticks to one pod. Try the demo with and without this flag.
"""

import argparse
import json
import threading
import time
from collections import Counter

import httpx
import numpy as np

from src.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30, help="seconds")
    parser.add_argument("--new-connections", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    payload = json.loads((PROJECT_ROOT / "examples" / "malignant.json").read_text())
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    if args.new_connections:
        # Ask the server to close the TCP connection after each response, so the next
        # request opens a new one (and the Service may route it to a different pod).
        headers["Connection"] = "close"
    latencies: list[float] = []
    served_by: Counter[str] = Counter()
    statuses: Counter[int] = Counter()
    lock = threading.Lock()
    deadline = time.time() + args.duration

    def worker() -> None:
        client = httpx.Client(base_url=args.url, headers=headers, timeout=10)
        while time.time() < deadline:
            start = time.perf_counter()
            try:
                r = client.post("/predict", json=payload)
                status, pod = r.status_code, r.headers.get("x-served-by", "?")
            except httpx.HTTPError:
                status, pod = 0, "connection-error"
            elapsed = time.perf_counter() - start
            with lock:
                latencies.append(elapsed)
                statuses[status] += 1
                served_by[pod] += 1
        client.close()

    print(f"Load test: {args.concurrency} threads for {args.duration}s against {args.url}")
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lat = np.array(latencies) * 1000
    print(f"\nrequests: {len(lat)}   throughput: {len(lat) / args.duration:.1f} req/s")
    print(
        f"latency ms  p50={np.percentile(lat, 50):.1f}  p95={np.percentile(lat, 95):.1f}  "
        f"p99={np.percentile(lat, 99):.1f}  max={lat.max():.1f}"
    )
    print(f"status codes: {dict(statuses)}")
    print("served by:")
    for pod, n in served_by.most_common():
        print(f"  {pod:<45} {n}")


if __name__ == "__main__":
    main()
