# Cancer MLOps

A production-style ML platform that predicts whether a breast tumour is **malignant or benign**
from 30 cell-nucleus measurements (Wisconsin Diagnostic Breast Cancer dataset, 569 patients).

The model is the small part. The project is about everything around it: validated data,
tracked and versioned experiments, a quality gate, a containerised API, monitoring,
drift detection, Kubernetes with autoscaling, and CI/CD.

> New here? Read [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) (what every folder does and why),
> then try [docs/DEMOS.md](docs/DEMOS.md) (run it, break it, watch it recover).

## Architecture

```
                    TRAINING (offline)                                SERVING (online)
 ┌───────────────────────────────────────────────┐
 │ data/raw/data.csv                             │        client ──HTTP──► Kubernetes Service (NodePort 30080)
 │   └─► validate (schema, nulls, ranges, dups)  │                              │ load-balances across READY pods
 │   └─► stratified split 80/20                  │                              ▼
 │   └─► tune LR, GB, MLP (5-fold CV)            │               ┌──────── Deployment: 3-10 pods (HPA) ────────┐
 │   └─► soft-voting + stacking ensembles        │               │ FastAPI  /predict  /predict/batch           │
 │   └─► select by CV ROC-AUC                    │               │          /health /ready /drift /metrics     │
 │   └─► threshold for recall >= 98% (OOF)       │               │ Pydantic validation -> model -> threshold   │
 │   └─► test once, bootstrap 95% CI            │               │ recent inputs -> KS drift test              │
 │   └─► quality gate                            │               └───────────────┬─────────────────────────────┘
 │        │                                      │                               │ /metrics (pull, every 5s)
 │        ├─► MLflow: runs, metrics, model files │                               ▼
 │        │   registry: version N, @champion     │                Prometheus (alert rules) ──► Grafana dashboard
 │        └─► artifacts/model/ (bundle)  ────────┼──► baked into Docker image cancer-api:<tag>
 └───────────────────────────────────────────────┘

 GitHub Actions:  CI (lint, types, tests, security, image smoke test) · CD (train → push image → deploy)
                  · weekly scheduled retraining with the same quality gate
```

## Results (real numbers from `python -m scripts.train`)

Hold-out test set = 114 patients (42 malignant), never used for training, tuning, model
selection or choosing the threshold.

| Model | CV ROC-AUC (5-fold, train) | Test accuracy @0.5 | Test recall @0.5 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | **0.996 ± 0.005** | 0.965 | 0.929 | 0.996 |
| Gradient Boosting | 0.995 ± 0.004 | 0.965 | 0.905 | 0.992 |
| Neural Network (MLP) | 0.989 ± 0.009 | 0.965 | 0.952 | 0.997 |
| Soft-voting ensemble | 0.994 ± 0.005 | 0.974 | 0.929 | 0.999 |
| Stacking ensemble | 0.995 ± 0.005 | 0.974 | 0.929 | 0.998 |

**Selected:** Logistic Regression (highest CV ROC-AUC). The decision threshold was lowered to
**0.207**, chosen on out-of-fold training predictions to reach about 98% recall. Final test results at that threshold:

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Recall (cancers caught) | **0.976** (41 of 42) | 0.921 – 1.000 |
| Precision | 0.932 | 0.848 – 1.000 |
| Accuracy | 0.965 | 0.930 – 0.991 |
| ROC-AUC | 0.996 | 0.986 – 1.000 |

Confusion matrix: 1 missed cancer (false negative), 3 false alarms (false positives).

**Honest reading:** all five models are within noise of each other on CV. The ensembles
did *not* clearly beat a well-tuned logistic regression on this small, clean dataset,
so the pipeline shipped the simpler model. With 114 test patients, one patient moves
accuracy by about 0.9%, so the confidence intervals matter more than the third decimal.

## Quick start

Requires Python 3.13, plus Docker Desktop for the container steps and [kind](https://kind.sigs.k8s.io/) for Kubernetes.

```powershell
# 1. Environment
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -c requirements.lock.txt

# 2. Train (logs to MLflow, registers the model, exports artifacts/model/)
python -m scripts.train
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://localhost:5000

# 3. Test + quality checks
pytest --cov
black --check src tests scripts; flake8 src tests scripts; mypy src scripts

# 4. Serve locally
uvicorn src.api.main:app --reload                        # http://localhost:8000/docs

# 5. Full stack in Docker: API + MLflow + Prometheus + Grafana
docker compose up -d --build

# 6. Kubernetes (kind): 3 replicas, HPA, probes
.\infrastructure\kubernetes\setup-kind.ps1               # http://127.0.0.1:30080/docs (not "localhost": see DEMOS.md)
```

Example request:
```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@examples/malignant.json"
# {"diagnosis":"malignant","malignant_probability":1.0,"threshold":0.2074,"model_version":"1"}
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | One patient → diagnosis, P(malignant), threshold, model version |
| `POST /predict/batch` | 1–1000 patients in one call |
| `GET /health` | Liveness: the process is alive |
| `GET /ready` | Readiness: the model is loaded and can take traffic |
| `GET /model/info` | Version, type, threshold, test metrics with CIs, MLflow run id, git sha |
| `GET /drift` | KS test of recent inputs vs. training data (Bonferroni-corrected) |
| `GET /metrics` | Prometheus metrics |

If `API_KEY` is set, `/predict*` requires the header `X-API-Key`. It is set in Kubernetes and off by default elsewhere.

## Project layout

```
configs/                 data + model settings (YAML), no code
src/data/                load -> validate -> split
src/models/              model definitions, metrics, training workflow, model bundle
src/api/                 FastAPI app, schemas, Prometheus metrics, settings, JSON logging
src/monitoring/          KS-test drift detector
scripts/                 entry points: train, simulate_drift, load_test
tests/                   unit (data, models, drift) + integration (API)
docker/                  Dockerfile (+ a deliberately bad one), Prometheus, Grafana config
docker-compose.yml       local 4-service stack
infrastructure/kubernetes/  kind cluster, Deployment, Service, ConfigMap, Secret, HPA, PDB
.github/workflows/       ci.yml, cd.yml, model_training.yml
docs/                    STUDY_GUIDE.md, DEMOS.md
```

## Limitations (stated plainly)

- **Small data.** There are 569 patients from a single source (1990s, University of Wisconsin). Results don't
  automatically transfer to other hospitals or imaging equipment. This is a teaching and portfolio
  dataset, not a clinical tool.
- **Drift is per replica.** Each API pod keeps its own window of recent inputs. A production
  version would send inputs to a shared store and run drift checks centrally.
- **Kubernetes is local (kind).** There is no cloud cluster, no Ingress controller and no TLS.
  The CD deploy job is wired up but skips unless a `KUBECONFIG_B64` secret for a real cluster exists.
- **The model is baked into the image.** That's simple and reproducible, but promoting a new model means a new
  image. The alternative is loading `models:/cancer-classifier@champion` from the MLflow registry at startup.
- **Retraining uses the same static dataset.** The scheduled workflow proves the mechanism.
  Real retraining needs a source of new labelled data.
- **Coverage (96%) excludes `src/models/train.py`.** The full training run is exercised by the
  training workflow and the CI image build, not by unit tests.
- **Secrets.** `secret.yaml` contains a demo key committed to git. That's acceptable for a local cluster, not for production.
