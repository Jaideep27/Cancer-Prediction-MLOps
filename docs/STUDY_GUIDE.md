# Study Guide: how this project works, folder by folder

Read this top to bottom with the code open next to it. Each "stop" names the files to open,
explains the idea in plain words, and ends with interview questions to test yourself.
Then do the matching exercise in [DEMOS.md](DEMOS.md). Running something and breaking it
teaches more than reading about it.

---

## The big picture in one analogy

Training a model in a notebook is **cooking one great meal at home**.
MLOps is **running a restaurant**: serving the same meal reliably to thousands of people,
noticing problems, and improving the recipe without closing the doors.

| Restaurant | This project | Folder |
|---|---|---|
| Checking ingredients on delivery | Data validation | `src/data/` |
| Developing the recipe | Training and tuning models | `src/models/` |
| The recipe notebook with every attempt written down | MLflow tracking and registry | `mlflow.db`, `mlruns/` |
| The waiter taking orders | FastAPI service | `src/api/` |
| Taste-testing before serving | Automated tests | `tests/` |
| A sealed lunchbox that tastes the same anywhere | Docker image | `docker/Dockerfile` |
| Kitchen cameras and a dashboard | Prometheus and Grafana | `docker/prometheus`, `docker/grafana` |
| Noticing customers' tastes have changed | Drift detection | `src/monitoring/` |
| The manager who replaces sick staff and calls in more at rush hour | Kubernetes | `infrastructure/kubernetes/` |
| The automatic health inspector at every delivery | GitHub Actions CI/CD | `.github/workflows/` |

### One prediction's journey (learn this, interviewers love it)
1. **Client sends the request.** A client sends `POST /predict` with 30 numbers.
2. **The Service picks a pod.** In Kubernetes, the **Service** picks one of the **ready** pods.
3. **The middleware starts a timer.** That's for the latency metric.
4. **The API key is checked.** This only happens if one is configured.
5. **Pydantic validates the input.** If anything is missing, negative, the wrong type or an unknown field, the response is `422` and the model never runs.
6. **The model scores the patient.** The 30 values become a DataFrame in the training column order. `predict_proba` gives P(malignant). If P ≥ 0.207 the answer is "malignant".
7. **The input is saved for drift checks.** It goes into the drift window, the recent-inputs buffer.
8. **The response goes back.** It includes the diagnosis, the probability, the threshold and the model version.
9. **The middleware records the outcome.** It counts the request and records its latency.
10. **Prometheus collects the numbers.** It scrapes `/metrics` every 5 seconds, Grafana draws them, and alert rules watch them.

---

## Stop 1: Data (`configs/data_config.yaml`, `src/data/`)

**Open:** `schema.py` → `load.py` → `validate.py` → `preprocess.py`

- **`schema.py`** is the *data contract*: the 30 feature names in a fixed order. The CSV has
  `concave points_mean` (with a space), so we rename it once to `concave_points_mean`. Everything
  downstream, including the API's JSON fields, uses the clean names.
- **`validate.py`** checks before the model sees anything: required columns, enough rows,
  numeric types, no missing values, no negative or infinite values (they are physical
  measurements), only `M` or `B` labels, and no duplicate patient ids. It collects **all**
  problems and reports them together.
  - *Analogy:* an airport security scanner. It's better to stop a bad bag at the door than find it in flight.
  - *Without it:* a single `NaN` or a typo'd label trains a broken model, and nobody notices.
- **`preprocess.py`** does a **stratified** split: 80% train, 20% test, with the same malignant
  ratio (37%) in both.
  - *Why stratify:* with only 212 malignant cases, a random split could, by bad luck, put too few
    in the test set and make the score meaningless.
  - `random_state=42` makes the split identical on every run, so runs are comparable.

### Data leakage (the most important ML concept in this project)
**Leakage** means information from the test set sneaks into training, so the test score looks better than reality.

- *Analogy:* a student who saw the exam questions while studying. A high score that proves nothing.
- **The classic leak** is scaling *before* splitting. `StandardScaler` learns the mean and spread of each
  feature. If it learns them from all 569 rows, the training process has already "seen" the test rows.
- **Our fix** is that the scaler is **inside a scikit-learn `Pipeline`** with the model (`src/models/build.py`).
  Whenever the pipeline is fitted, whether on the full train set or on one cross-validation fold,
  the scaler only learns from that training data. `preprocess.py` deliberately contains **no** scaling.
- **Other leaks we avoid:**
  - choosing the model using the test set (we use CV);
  - choosing the threshold on the test set (we use out-of-fold train predictions);
  - duplicate patients in both train and test (validation checks ids).

**Test yourself:** Why fit the scaler on training data only? What does stratification protect against? Name two leaks besides scaling.

---

## Stop 2: Models (`configs/model_config.yaml`, `src/models/`)

**Open:** `build.py` → `evaluate.py` → `train.py` → `scripts/train.py`

### The three base models, in plain words
| Model | How it thinks | Needs scaling? |
|---|---|---|
| **Logistic Regression** | Gives each feature a weight, adds them up, squashes the sum into a 0–1 probability. Simple and explainable. | Yes |
| **Gradient Boosting** | Builds many small decision trees, one after another. Each new tree fixes the previous trees' mistakes. | No, trees split on thresholds |
| **Neural Network (MLP)** | Layers of weighted sums with non-linear "bends" in between. Can learn curved boundaries. | Yes, very sensitive to scale |

We use scikit-learn's `MLPClassifier`, not TensorFlow. TensorFlow would add about 1 GB to the image and has Python 3.13
friction. For 30 numeric columns, a small MLP is the right tool.

### Ensembles: why combine models?
Different models make **different mistakes**. Combining them can cancel some mistakes out, like asking three doctors instead of one.
- **Soft voting** (`VotingClassifier`) averages the three probabilities. It's simple and robust.
- **Stacking** (`StackingClassifier`) trains a small logistic regression to *learn* how much to trust each model.
  It's trained on **out-of-fold** predictions, so it never sees a model's guess on data that model was trained on.

### Cross-validation (CV) and tuning
- **5-fold stratified CV:** cut the 455 training rows into 5 parts. Train on 4 and score on the 5th, rotating
  5 times. That gives 5 scores, so you see the average **and the spread**. One split can be lucky; five are harder to fool.
- **`RandomizedSearchCV`:** tries 15 random hyperparameter combinations per model (a full grid would be hundreds),
  each scored with 5-fold CV, and keeps the best. Hyperparameters are settings you choose rather than ones the model learns, such as
  `C` (how strongly to keep weights small) or `max_depth` (tree size).

### The metrics (positive = malignant)
| Metric | Question it answers | Cancer meaning |
|---|---|---|
| **Accuracy** | What fraction of all predictions were right? | Can mislead: always saying "benign" scores 63% |
| **Precision** | Of the patients we flagged as cancer, how many really had it? | Low precision means false alarms and unnecessary biopsies |
| **Recall** (sensitivity) | Of all real cancers, how many did we catch? | Low recall means **missed cancers** |
| **F1** | The balance of precision and recall (harmonic mean) | One number when both matter |
| **ROC-AUC** | If you pick one random cancer and one random benign case, how often does the model score the cancer higher? | Measures ranking quality and **doesn't depend on the threshold** |

**Why recall matters most here:** the two errors have very different costs.
- A false alarm costs an extra test and some worry.
- A missed cancer can cost a life.

So we deliberately trade some precision for recall.

### The decision threshold
The model outputs a probability, and the threshold turns it into a yes or no. The default is 0.5. We lowered it to
**0.207**, so any case with more than about a 21% chance of being malignant gets flagged.
- **Chosen on out-of-fold training predictions** (`cross_val_predict`) to reach 98% recall, **never on the test set**.
- **Capped at 0.5:** we only ever *lower* the bar for "malignant", never raise it.
- **The effect on the test set:** recall went from 0.929 at 0.5 to **0.976** at 0.207, which is 1 missed cancer instead of 3. The cost was 3 false alarms instead of 1.

### Model selection and the honest result
- The pipeline picks the candidate with the best **CV ROC-AUC**, which is computed on train data only.
- **Logistic Regression won** with 0.9958. The others were 0.9888 to 0.9951.
- Those differences are **smaller than the fold-to-fold spread** (±0.005), so they are noise.
- **What to say:** "I built and evaluated a hybrid ensemble. On this dataset it didn't significantly beat a
  well-tuned logistic regression, so the selection step chose the simpler, more explainable model.
  That's exactly what an automated selection step is for." This is a *stronger* interview answer than "97% ensemble".

### Overfitting and defending results on 569 rows
**Overfitting** means memorising the training data instead of learning the pattern. It shows up as great training scores and poor scores on new data.

Our defences:
1. **A held-out test set**, used once, at the end.
2. **CV mean ± std** instead of a single number.
3. **Regularisation:** `C` for LR, `alpha` for the MLP, shallow trees and `subsample` for GB, and early stopping for the MLP.
4. **Bootstrap 95% confidence intervals** on the test metrics. For example, accuracy 0.965 has the interval [0.930, 0.991].

Honest caveat: CV scores after tuning on the same folds are slightly optimistic. *Nested CV*, an outer CV loop
around the tuning, would fix that, and it's a good "what would you improve?" answer.

### Quality gate (`promotion:` in the config)
A new model becomes the "champion" only if test recall is at least 0.95 and ROC-AUC is at least 0.97.
`scripts/train.py` exits with code 1 otherwise, which **fails the CI/CD pipeline**, so a bad model can't ship.

**Test yourself:** Why is accuracy misleading here? What's the difference between voting and stacking?
Why choose the threshold on out-of-fold predictions? How would you defend results from 569 rows?

---

## Stop 3: MLflow (`src/models/train.py`, `setup_mlflow` and `evaluate_and_log`)

See it: run `mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://localhost:5000.

- **Experiment** = a folder of related attempts (`cancer-classification`).
- **Run** = one attempt, with its **params** (settings), **metrics** (scores), **artifacts** (files, including
  the saved model) and **tags**. Each training session is one *parent run* with 5 *child runs*, one per candidate.
- **Tracking vs registry:**
  - *Tracking* is the **lab notebook**: every experiment, good or bad, recorded so you can compare them and reproduce them.
  - *Registry* is the **shelf of approved products**: named models (`cancer-classifier`) with numbered
    **versions** (1, 2, 3…) and **aliases** like `@champion`, which are movable labels that point to one version.
- **Rollback** = move `@champion` back to the previous version. That's one command, with no retraining.
- **Why a team needs it:** "Which settings produced the model in production? Who trained it, on what data, at which git commit?"
  Without it, the answer is a spreadsheet, or nobody knows.
- **skops vs pickle:** loading a pickle file can run arbitrary code. MLflow now saves scikit-learn models with
  *skops*, which refuses unknown types unless you explicitly trust them. We list exactly the 4 scikit-learn internal
  types our models contain (`SKOPS_TRUSTED_TYPES`), and nothing else.
- **The bundle (`src/models/bundle.py`):** the champion is also exported to `artifacts/model/` (model + metadata +
  reference data). The API loads that folder, so the serving image doesn't need MLflow at all.

**Test yourself:** Tracking vs registry? How do you roll back a bad model? What metadata would you want for an audit?

---

## Stop 4: The API (`src/api/`)

**Open:** `schemas.py` → `settings.py` → `main.py` → `metrics.py`

- **Pydantic schemas** check the input before your code runs. All 30 fields are required, each is a float ≥ 0,
  and `extra="forbid"` rejects typos like `radius_mea`. Bad input gets an automatic `422` with a clear message.
  - *Analogy:* a bouncer checking IDs at the door, so the bar staff never deal with it.
- **The model loads at startup (`lifespan`), not per request.** Loading takes over 100 ms; predicting takes about 1 ms.
- **Fail fast:** if the model can't load, the app **refuses to start**.
  - *The original repo* caught the error and started anyway. That gives you a server that looks "up" but returns errors on every request.
  - A clearly dead server is better, because Kubernetes then sees it and won't send it traffic.
- **`def` instead of `async def` for `/predict`:** prediction is CPU work. FastAPI runs plain `def` endpoints in a
  thread pool, so one slow prediction doesn't block every other request on the event loop.
- **Liveness (`/health`) vs readiness (`/ready`):**
  - *Liveness* asks "is the process alive?" If it fails, the container is **restarted**.
  - *Readiness* asks "can I serve traffic right now?" If it fails, the pod is **removed from load balancing** but not restarted.
- **Model versioning in responses:** every prediction says which model version made it. When a doctor asks
  "why did it say that last Tuesday?", you can answer.
- **API key:** off by default. When `API_KEY` is set, `/predict*` requires the `X-API-Key` header.
  `secrets.compare_digest` makes the check take the same time whether the guess is close or not.
- **`X-Served-By` header** = the hostname (the pod name in Kubernetes). It shows load balancing happening.

**Test yourself:** What happens if the client sends a negative radius? Why load the model at startup? Liveness vs readiness?

---

## Stop 5: Tests (`tests/`)

43 tests, 96% coverage of `src/` (excluding `train.py`). Run them with `pytest --cov`.

| Test file | Protects against |
|---|---|
| `unit/test_data.py` | Schema changes, NaNs, negative values, bad labels and duplicate ids slipping through; a split that's not stratified, overlaps, or isn't reproducible |
| `unit/test_models.py` | Broken model code; the scaler moving outside the pipeline (leakage); wrong metric maths; a threshold that misses its recall target; a corrupt model bundle |
| `unit/test_drift.py` | False alarms on normal data; missing real drift; a forgotten Bonferroni correction |
| `integration/test_api.py` | Broken endpoints; 4 kinds of invalid input getting through; batch limits; auth bypass; missing metrics; **a server starting without a model** |

- **Unit vs integration:** a unit test checks one piece in isolation. An integration test checks pieces working together:
  here, the real FastAPI app with a real model, called over HTTP in-process.
- **`conftest.py`** trains a tiny model into a temp folder, so the tests run on a fresh CI machine without full training.

**Test yourself:** What's the difference between a unit and an integration test? Why don't the API tests use the real trained model?

---

## Stop 6: Docker (`docker/Dockerfile`, `docker/Dockerfile.bad`, `.dockerignore`)

### The words
- **Image:** a read-only template containing the OS files, Python, libraries, your code and the model. *Analogy:* a recipe and its sealed ingredients.
- **Container:** a running instance of an image. *Analogy:* a meal cooked from that recipe. You can cook many from one recipe.
- **Layer:** each `FROM`, `RUN` or `COPY` instruction makes one layer, like a stack of transparent sheets. Docker **caches** layers:
  if an instruction and its inputs haven't changed, the old layer is reused.
- **Volume:** storage that lives **outside** the container, so data survives when the container is deleted (MLflow's database, for example).
- **Registry:** a store for images, such as Docker Hub or GitHub Container Registry.

### Containers vs virtual machines
- A **VM** runs a whole guest operating system on virtual hardware. It's heavy: GBs, and a minute to boot.
- A **container** shares the host's kernel, the core of the OS. It's only an isolated *process* with its own filesystem view. It's light: MBs, and starts in under a second.
- *Analogy:* a VM is a separate house; a container is an apartment in a shared building.
- **On Windows:** Docker Desktop runs a small Linux VM (WSL2), and your Linux containers run inside it.

### The Dockerfile, line by line (good vs bad)
| Choice | Good (`Dockerfile`) | Bad (`Dockerfile.bad`) | Why it matters |
|---|---|---|---|
| Base image | `python:3.13-slim` | `python:latest` | Pinned means reproducible; slim saves about 1 GB |
| Build stages | builder + runtime | single stage | The final image keeps only what runs |
| Dependency order | copy requirements **first**, install, **then** copy code | `COPY . .` then install | Code changes reuse the cached install layer: seconds instead of minutes |
| What's installed | `requirements-serve.txt` with `-c requirements.lock.txt` | dev requirements, unpinned | No MLflow or pytest in production; exact library versions |
| User | `USER 10001` (non-root) | root | Someone breaking into the app doesn't get root |
| Health check | `HEALTHCHECK` on `/health` | none | `docker ps` shows healthy or unhealthy; Compose can wait for it |
| Start command | exec form `["uvicorn", ...]` | shell form | uvicorn is PID 1 and receives SIGTERM, so it shuts down gracefully |

- **Why the model is baked into the image:** image tag = model version. You can reproduce exactly what was running on any day.
- **Port mapping:** `-p 8000:8000` means the host's port 8000 forwards to the container's port 8000. Without it, the container's port is unreachable from your laptop.

**Test yourself:** Image vs container? Why copy requirements before code? Why non-root? Container vs VM?

---

## Stop 7: Docker Compose (`docker-compose.yml`)

Compose starts several containers as one app with one command, `docker compose up -d --build`.

- **Networking by name:** Compose creates a private network with its own DNS (the system that turns names into addresses). Prometheus reaches the API at
  `http://api:8000` and Grafana reaches Prometheus at `http://prometheus:9090`, using service names, not IPs.
  - *Analogy:* calling colleagues by name on the office phone system instead of memorising extensions.
- **Volumes** (`mlflow-data`, `prometheus-data`, `grafana-data`): `docker compose down` deletes containers but keeps
  volumes. Adding `-v` deletes the volumes too.
- **Environment variables:** `API_KEY: ${API_KEY:-}` comes from a `.env` file, with an empty default. The same image behaves
  differently per environment without being rebuilt.
- **`depends_on` with `condition: service_healthy`:** Prometheus starts only after the API's HEALTHCHECK passes.
  Plain `depends_on` only waits for the container to *start*, not to be *ready*.
- **`restart: unless-stopped`:** if a process crashes, Docker restarts it. This is the simple version of what Kubernetes does.

**Test yourself:** How does Prometheus find the API? What survives `docker compose down`? Why `service_healthy`?

---

## Stop 8: Monitoring (`src/api/metrics.py`, `docker/prometheus/`, `docker/grafana/`, `src/monitoring/drift.py`)

### Prometheus
- **Metric:** a named number over time, with labels, such as `http_requests_total{path="/predict",status="200"}`.
- **Types:**
  - **Counter** only goes up, like requests served. You look at its *rate*.
  - **Histogram** counts observations into buckets (≤5 ms, ≤10 ms, …), which is what lets you compute percentiles.
  - **Gauge** goes up and down, like the drift share.
- **Scraping (pull model):** Prometheus calls `GET /metrics` every 5 seconds. Your app just keeps numbers in memory and doesn't need to know Prometheus exists.
- **Label cardinality:** labels must come from a small fixed set. We use the route *template* `/predict`, never
  user input. Each unique label combination is a new time series, and millions of them crash Prometheus.

### PromQL you should know (try them at http://localhost:9090)
```
rate(http_requests_total[1m])                                   # requests per second, per series
sum by (path) (rate(http_requests_total[1m]))                   # per endpoint
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))   # error ratio
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))   # p95 latency
```
**p95 latency** = the time under which 95% of requests finish. Averages hide slow outliers; percentiles show them.

### Grafana and alerts
- **Grafana** queries Prometheus and draws the dashboard, which covers the request rate, p50/p95/p99 latency, errors, predictions by class, drift and model version.
- **Provisioning** means the data source and dashboard are created from files at startup, so nothing has to be clicked by hand.
- **Alerts** live in `alert_rules.yml`: API down, error rate over 5%, p95 over 250 ms, and data drift.
  - The `for:` clause means a condition must hold for a while before the alert fires. This avoids paging someone for a single blip.
  - Grafana shows these rules under Alerting as data-source-managed rules.
  - Sending alert notifications (email, Slack) would need Alertmanager, which is **not built here**.

### Data drift and the KS test
- **Drift** means production inputs stop looking like the training data: a new scanner, a different hospital, a population change.
  The model was never taught that region, so **accuracy decays silently**, with no errors and no crashes.
- **The KS test** compares two samples of one feature, the training reference and recent requests. It measures the biggest gap between their
  cumulative distributions. A small p-value means "these probably come from different distributions".
- **Bonferroni correction:** testing 30 features at p<0.05 gives about 1.5 false alarms per check by chance alone. So each
  feature needs p < 0.05/30 = 0.00167. The dataset-level alarm fires when at least 10% of features drift.
  - You saw this in practice: on normal traffic, 1 feature was flagged by chance, and the dataset alarm correctly stayed off.
- **Limits:**
  - The test is univariate: it can't see correlated shifts across features.
  - It detects *input* drift, not whether predictions are wrong (that needs true labels later, which is *concept drift*).
  - The window is per replica.

**Test yourself:** Counter vs histogram? Why percentiles, not averages? Why does a model decay? Why Bonferroni?

---

## Stop 9: Kubernetes (`infrastructure/kubernetes/`)

**Kubernetes (K8s)** keeps a desired state true. You declare "3 copies of this container, each with these
resources", and it keeps working to make reality match. That loop is called *reconciliation*.
- *Analogy:* a thermostat. You set 21 °C, and it keeps acting until the room is 21 °C.

| Object | Plain words | Our file |
|---|---|---|
| **Cluster** | All the machines Kubernetes manages | `kind-config.yaml` (3 Docker containers acting as machines) |
| **Node** | One machine. The *control plane* is the brain; *workers* run your apps | 1 control plane + 2 workers |
| **Pod** | The smallest unit: one or more containers sharing an IP. Disposable | created by the Deployment |
| **ReplicaSet** | Keeps exactly N identical pods alive | created by the Deployment |
| **Deployment** | Manages ReplicaSets, which enables rolling updates and rollback | `deployment.yaml` |
| **Service** | A stable name and IP in front of changing pods; balances load across **ready** pods | `service.yaml` |
| **ConfigMap** | Non-secret settings, injected as environment variables | `configmap.yaml` |
| **Secret** | Sensitive settings (base64-encoded, **not encrypted** by default) | `secret.yaml` |
| **HPA** | Adds or removes pods based on CPU usage | `hpa.yaml` |
| **PodDisruptionBudget** | "Keep at least 2 pods during maintenance" | `pdb.yaml` |
| **PersistentVolume / PVC** | Disk that outlives pods | *not needed*: the API is stateless and the model is in the image |
| **Ingress / Gateway** | HTTP routing by hostname or path, plus TLS, from outside | *not built*; we use a NodePort |
| **Namespace** | A folder inside the cluster | `namespace.yaml` |

### Key ideas in `deployment.yaml`
- **Requests vs limits:**
  - *Request* is what the scheduler reserves on a node (250m = a quarter of a CPU core), and it's the HPA's 100% mark.
  - *Limit* is the ceiling. CPU above it is **throttled** (slowed). Memory above it gets the container **OOMKilled** (killed for running out of memory).
- **Three probes:**
  - *startup* gives the app up to 60 seconds to load the model before the other probes start.
  - *readiness* failing takes the pod out of the Service, so it gets no traffic.
  - *liveness* failing kills and restarts the container.
- **Rolling update:** `maxSurge: 1, maxUnavailable: 0` means one new pod starts, becomes ready, and only then is an old one
  removed. That's zero downtime. If a new version never becomes ready, the old pods keep serving.
- **Rollback:** `kubectl rollout undo` switches back to the previous ReplicaSet (`revisionHistoryLimit: 5`).
- **Self-healing:** delete a pod and the ReplicaSet sees 2 instead of 3 and creates a new one within seconds.
- **`preStop: sleep 5`:** on shutdown, wait 5 seconds so the Service stops routing to the pod *before* the app exits. No dropped requests.
- **Security:** non-root user 10001, read-only root filesystem, all Linux capabilities dropped, no privilege escalation.
- **topologySpreadConstraints** spread the replicas across both workers, so one node dying doesn't take all of them with it.

### HPA maths
desired replicas = ceil(current replicas × current CPU% ÷ target CPU%).
- *Example:* 3 pods at 120% of their request, with a 60% target: ceil(3 × 120 ÷ 60) = 6 pods.
- It reads CPU from **metrics-server**, which `setup-kind.ps1` installs.
- It waits 60 seconds of low load before scaling down (`stabilizationWindowSeconds`), so it doesn't flap up and down.

### Why Kubernetes and not just Docker Compose?
- **Compose is one machine.** If that machine dies, everything is down. Compose has no autoscaling, no rolling updates with
  readiness gating, and no scheduling across machines.
- **Kubernetes** runs across many machines, self-heals, autoscales, does zero-downtime updates and rollbacks, and offers a
  standard API for every cloud.
- **The honest trade-off:** Kubernetes is complex. For one small API with low traffic, Compose or a managed container
  service such as Cloud Run or ECS can be the *better* choice.

**Test yourself:** What happens when you delete a pod? Readiness vs liveness? Requests vs limits? How does the HPA decide?

---

## Stop 10: CI/CD (`.github/workflows/`)

- **Trigger:** what starts a workflow, such as a `push`, a `pull_request`, a version tag, a `schedule` (cron) or a manual `workflow_dispatch`.
- **Job:** a group of steps on a fresh virtual machine. Jobs run **in parallel** unless `needs:` orders them.
- **Secret:** an encrypted value stored in GitHub settings, like `KUBECONFIG_B64`. It's never printed in logs.
  `GITHUB_TOKEN` is created automatically for each run.
- **Caching:** `cache: pip` reuses downloaded packages between runs, and `cache-from: type=gha` reuses Docker layers. Both make runs faster.

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | every push and PR to main | black, flake8, mypy · pytest with coverage ≥80% · pip-audit + bandit · train → build image → run it → smoke-test real predictions |
| `cd.yml` | tag `v*.*.*` or manual | train with the quality gate → push the image to ghcr.io → **manual approval** → rolling update with automatic `rollout undo` on failure |
| `model_training.yml` | every Monday 06:00 UTC or manual | retrain → the gate fails the run if the model is worse → upload the model and metrics |

- **Continuous delivery vs deployment:**
  - *Delivery:* every change is *ready* to release, and a human presses the button. That's our `environment: production` approval.
  - *Deployment:* every passing change goes live automatically, with no human step.
- **Honest note:** the deploy job needs an internet-reachable cluster. Your kind cluster isn't one, so the job
  skips. The workflows haven't been run on GitHub yet: push the repo to see them run.

**Test yourself:** What does each workflow protect? Delivery vs deployment? How does a bad model get blocked?

---

## What changed vs the original repo, and why

| Original | Rebuild | Reason |
|---|---|---|
| Model-load errors swallowed at startup | Fail fast, plus a `/ready` probe | A "healthy" server with no model is the worst failure mode |
| No visible `/metrics` | Prometheus counters, histograms and gauges | Prometheus and Grafana had nothing real to scrape |
| Root user, `requests`-based healthcheck, Python 3.9 | User 10001, standard-library healthcheck, Python 3.13 | Security, fewer dependencies, a supported Python version |
| mlxtend voting + TensorFlow | scikit-learn `VotingClassifier`/`StackingClassifier` + `MLPClassifier` | One library, a smaller image, all models inside Pipelines |
| No visible CV, tuning or leakage control in the trainer | Pipelines, stratified CV, random search, OOF threshold, bootstrap CIs | Defensible numbers |
| KS test at p<0.05 × 30 features | Bonferroni + dataset-level threshold + minimum samples | Stops false alarms |
| Reference data passed in manually | Saved with the model bundle (`reference.csv`) | The baseline always matches the model |
| README claimed 97% | README reports measured numbers with confidence intervals | Honesty |
| Terraform placeholder | Removed | Don't ship empty folders that imply work |

---

## Talking about it honestly

| You **can** say | You **cannot** say |
|---|---|
| "I built and ran it locally: Docker Compose and a 3-node kind cluster" | "Deployed to production / the cloud" |
| "Recall 97.6% on a 114-patient hold-out, CI 92–100%" | "97% accurate" with no context |
| "The ensemble didn't beat logistic regression, so the pipeline chose LR" | "The hybrid ensemble is my best model" |
| "CI/CD workflows are written; the deploy step needs a real cluster" | "Fully automated deployment pipeline" (unless you push to GitHub and it runs) |
| "96% coverage of src, excluding the training script" | ">80% coverage" as a vague claim |
| "Drift detection works per replica; centralising it is the next step" | "Production drift monitoring" |

---

## Interview question bank (answer out loud, then check the stop)

1. Walk me through what happens when a request hits `/predict`. *(Big picture)*
2. How did you prevent data leakage? *(Stop 1)*
3. Why is recall more important than precision here, and how did you raise it? *(Stop 2)*
4. Your test set is 114 rows. How confident are you in 97.6% recall? *(Stop 2: bootstrap CI)*
5. Why didn't you ship the ensemble? *(Stop 2)*
6. MLflow tracking vs registry? How do you roll back a bad model? *(Stop 3)*
7. What stops a worse model from reaching production? *(Stop 2 quality gate, Stop 10)*
8. Why load the model at startup, and what if loading fails? *(Stop 4)*
9. Liveness vs readiness: give a case where one fails and the other passes. *(Stop 4, Stop 9)*
10. Explain your Dockerfile's layer order. *(Stop 6)*
11. How does Prometheus find your API in Compose? In Kubernetes? *(Stop 7, Stop 9)*
12. Counter vs histogram? Write a p95 latency query. *(Stop 8)*
13. What is drift, how do you detect it, and what are the limits of the KS test? *(Stop 8)*
14. What happens when you delete a pod? When a node dies? *(Stop 9)*
15. How does the HPA decide to scale? What are requests vs limits? *(Stop 9)*
16. How would you scale this 10×? What breaks first? *(Answer: one process per pod means CPU-bound prediction. Scale pods with the HPA; batch requests; move drift detection to a shared async job; add a Prometheus scraper per pod with service discovery; load-test to find the knee.)*
17. Why Kubernetes instead of just Docker Compose? *(Stop 9)*
18. Continuous delivery vs deployment? *(Stop 10)*
19. What would you improve next? *(nested CV, centralised drift + label feedback loop, MLflow-registry loading or a model server, Ingress + TLS, Alertmanager, a real cloud cluster with Terraform)*
