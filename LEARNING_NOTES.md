# Learning Notes: Cancer MLOps Rebuild

How this was built: Step 0 was done step by step. After that, the whole project was built in one go
and verified by running it. Learning now happens by reading the finished system
([docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md)) and running and breaking it ([docs/DEMOS.md](docs/DEMOS.md)).
This file records **what was actually run and observed**, plus the problems hit along the way.

---

## Step 0: Setup

**What I built**
- Project root `C:\Resume_Projects\Cancer_MLOPs`, git repo initialised, Python 3.13 venv in `venv/`.
- Folder layout: `src/{data,models,api,monitoring}`, `configs`, `tests/{unit,integration}`, `scripts`, `docker`, `infrastructure/kubernetes`, `.github/workflows`, `data/raw`, `docs`, `examples`.
- Dataset: `data/raw/data.csv`, 569 patients (212 malignant, 357 benign), 30 features.

**Key concepts**
- **venv:** each project gets its own library versions. Proven in practice: outside the venv `mlflow` was 3.6.0 (Anaconda), inside it was 3.16.1.
- **Requirements files:**
  - `requirements-serve.txt`: the API only (goes into Docker).
  - `requirements.txt`: serve + MLflow (training).
  - `requirements-dev.txt`: + test and lint tools.
  - `requirements.lock.txt`: exact versions, used as `-c` constraints in Docker and CI.
- **Layout:** separation of concerns; dependencies point one way (api → models → data).

---

## Verified results (all run on this machine)

| Area | What was run | Observed |
|---|---|---|
| Data | `validate_and_clean` on clean and corrupted data | All problems reported at once (missing values, negatives, unknown labels) |
| Training | `python -m scripts.train` (~80 s) | Logistic Regression selected; threshold 0.207; **test recall 0.976 [0.921–1.0]**, accuracy 0.965, ROC-AUC 0.996; 1 missed cancer, 3 false alarms |
| Honest finding | 5 candidates compared by CV ROC-AUC | Ensembles (0.994–0.995) did **not** beat LR (0.996); differences are within fold noise (±0.005) |
| MLflow | Local `mlflow.db` + Compose server | 5 child runs per training; registry `cancer-classifier` v1 → v2, alias `@champion` |
| Tests | `pytest --cov` | 43 passed, 96% coverage of `src/` (excluding `train.py`) |
| Quality | black, flake8, mypy, bandit, pip-audit | All clean; no known vulnerabilities in serving deps |
| Docker | Build + run + `exec` | Image 161 MB compressed / 704 MB on disk; healthy in ~6 s; runs as user `app` (10001); only `src/` + `artifacts/` inside |
| Compose | `docker compose up -d --build` | 4 services up; Prometheus target UP; 4 alert rules loaded; Grafana datasource and dashboard provisioned automatically |
| Load (Compose) | 10 threads × 30 s | 3,819 requests, 0 errors, p95 ≈ 111 ms, 1 container |
| Drift | `scripts.simulate_drift` | Normal: 1/30 features flagged by chance, **no** dataset alarm. Shifted: 11/30 flagged, alarm on. `DataDriftDetected` fired after ~45 s |
| Kubernetes | `setup-kind.ps1` | 3 nodes Ready (v1.37.0); 3 pods across 2 workers; API key enforced (401 without, 200 with) |
| Self-healing | Deleted a pod | Replacement Ready in ~6 s |
| Load balancing | `load_test --new-connections` | Even split, e.g. 495 / 466 / 458 requests |
| Autoscaling | 30 threads × 120 s | 3 → 7 → 10 pods within 50 s; back to 3 at t+210 s; **0 errors in 23,835 requests** |
| Rolling update + rollback | Under live traffic | **0 errors in 11,413 requests** |
| Bad image | `set image ...:does-not-exist` | New pod `ImagePullBackOff`; old pods kept serving; undo fixed it |
| Broken readiness probe | Probe path `/nope` | New pod `0/1 Running`, never got traffic; old pods kept serving |
| Out of memory | Limit 60Mi (app needs ~125Mi) | `OOMKilled`, exit code 137, `CrashLoopBackOff`; old pods kept serving |

---

## Problems hit and what they taught

| Problem | Root cause | Fix / lesson |
|---|---|---|
| MLflow refused to save GB / MLP / ensemble models | MLflow saves with **skops**, which blocks unknown types (pickle can run arbitrary code) | Trust exactly the 4 scikit-learn internal types, found with `skops.io.get_untrusted_types()` |
| `docker pull` failing with `httpReadSeeker ... EOF` | Docker Hub's download servers dropping the connection on this network | Pull from `quay.io` / `mirror.gcr.io`, then `docker tag`; retry for kind's node image. *Lesson: Docker Hub is a single point of failure, so teams mirror images* |
| `setup-kind.ps1` crashed on "No kind clusters found" | Windows PowerShell 5 treats stderr output as a terminating error under `ErrorActionPreference=Stop` | Use `Continue` and check `$LASTEXITCODE` after each command |
| Every request to kind took ~2 s | Windows `localhost` tries IPv6 first; kind's port mapping is IPv4-only | Use `127.0.0.1:30080` (measured 220 ms vs 7 ms with curl, ~2 s with Python) |
| The autoscaler never scaled during the first load test | The load script rebuilt its HTTP client per request (5.6 req/s) | Send `Connection: close` instead, which gave 142–199 req/s |
| Out-of-memory demo rejected | A memory request (256Mi) can't exceed the limit (60Mi); the API server validates every change | Lower the request too |
| `rollout undo` restored a **broken** version | `undo` = "back one revision"; the rejected change made no revision, so undo went to the version before | Use `--to-revision=N`, or `kubectl apply -k` from git (the YAML in git is the source of truth) |
| Model lineage pointed to the wrong MLflow server | Last training run had `MLFLOW_TRACKING_URI` set to the Compose server | Retrained locally; the bundle's run id now exists in `mlflow.db` |

---

## Key commands

```powershell
python -m scripts.train                                       # train + register + export
mlflow ui --backend-store-uri sqlite:///mlflow.db             # http://localhost:5000
pytest --cov                                                  # tests
uvicorn src.api.main:app --reload                             # local API
docker compose up -d --build ; docker compose ps ; docker compose down
.\infrastructure\kubernetes\setup-kind.ps1                    # kind cluster + app
kubectl -n cancer-mlops get pods -o wide ; kubectl -n cancer-mlops get hpa -w
kubectl -n cancer-mlops rollout history|undo|status deployment/cancer-api
kubectl apply -k infrastructure/kubernetes                    # back to the declared state
kind delete cluster --name cancer-mlops                       # free the RAM
```

## Interview Q&A
The full question bank is at the end of [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md). Three answers from Step 0:
1. *Why a virtual environment?* It isolates a project's dependencies so each project gets its own library versions. That prevents conflicts and makes the setup reproducible.
2. *Why several requirements files and a lock file?* Runtime-only dependencies go in the image; dev tools stay out of production; the lock file pins exact versions, including indirect ones, so builds are reproducible.
3. *Why split src / tests / configs / scripts?* Separation of concerns, one-way dependencies, and Docker/CI can pick only what they need.
