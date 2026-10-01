"""Show drift detection working against a RUNNING API.

    python -m scripts.simulate_drift --url http://localhost:8000

Phase 1: send realistic patients (held-out test rows + tiny noise) -> GET /drift -> expect no drift.
Phase 2: send "drifted" patients (a recalibrated scanner reports sizes 40% larger,
         texture noisier) -> GET /drift -> expect drift, with the shifted features flagged.

Note: each API replica keeps its OWN window of recent inputs. Against Kubernetes
with 3 replicas, run this through `kubectl port-forward` to a single pod, or
raise --n so every replica receives enough samples.
"""

import argparse

import httpx
import numpy as np
import pandas as pd

from src.data.preprocess import load_dataset

SIZE_FEATURES = [
    f"{base}_{stat}" for base in ("radius", "perimeter", "area") for stat in ("mean", "se", "worst")
]


def send(client: httpx.Client, rows: pd.DataFrame, batch: int = 50) -> None:
    for start in range(0, len(rows), batch):
        chunk = rows.iloc[start : start + batch].to_dict(orient="records")
        r = client.post("/predict/batch", json={"instances": chunk})
        r.raise_for_status()


def show(client: httpx.Client, title: str) -> None:
    report = client.get("/drift").json()
    print(f"\n=== {title} ===")
    if report["status"] != "ok":
        print(report)
        return
    print(
        f"window={report['n_current']}  drifted features={report['n_drifted']}/30  "
        f"dataset_drift={report['dataset_drift']}  "
        f"(per-feature alpha={report['per_feature_alpha']:.5f}, Bonferroni)"
    )
    top = sorted(report["features"], key=lambda f: f["p_value"])[:6]
    for f in top:
        flag = "DRIFT" if f["drifted"] else "ok"
        print(f"  {f['feature']:<24} KS={f['ks_statistic']:.3f}  p={f['p_value']:.2e}  {flag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500, help="rows per phase (= API drift window)")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    _, X_test, _, _ = load_dataset()
    normal = X_test.sample(args.n, replace=True, random_state=0).reset_index(drop=True)
    normal = normal * rng.normal(1.0, 0.02, normal.shape)  # small measurement noise

    drifted = normal.copy()
    drifted[SIZE_FEATURES] *= 1.4
    drifted["texture_mean"] *= rng.normal(1.3, 0.1, len(drifted))
    drifted = drifted.clip(lower=0)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as client:
        send(client, normal)
        show(client, "Phase 1: normal traffic")
        send(client, drifted)  # fills the whole window, pushing the normal rows out
        show(client, "Phase 2: drifted traffic (sizes x1.4, texture x1.3)")
    print("\nCheck Grafana's drift panel and http://localhost:9090/alerts (DataDriftDetected).")


if __name__ == "__main__":
    main()
