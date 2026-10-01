# Demos: run it, break it, watch it recover

All commands are PowerShell, run from the project root with the venv active
(`.\venv\Scripts\Activate.ps1`). Use `curl.exe`, not `curl`: in Windows PowerShell, `curl` is an alias for a different command.

---

## 1. Data validation

```powershell
python -c "from src.data.load import load_raw; from src.data.validate import validate_and_clean; print(validate_and_clean(load_raw()).shape)"
# (569, 31)  -> 30 features + target
```
**Break it.** Feed it corrupted data and watch every problem get reported at once:
```powershell
python -c "from src.data.load import load_raw; from src.data.validate import validate_and_clean; d=load_raw(); d.loc[0,'area_mean']=None; d.loc[1,'diagnosis']='X'; d.loc[2,'radius_mean']=-3; validate_and_clean(d)"
```

## 2. Training and MLflow

```powershell
python -m scripts.train                                   # about 80 seconds
mlflow ui --backend-store-uri sqlite:///mlflow.db         # http://localhost:5000
```
In the UI:
1. Open **Experiments â†’ cancer-classification**.
2. Expand the latest parent run, select the 5 child runs, and click **Compare**.
3. Open **Models â†’ cancer-classifier** and see each version and the `@champion` alias.

**Break it.** Make the quality gate impossible to pass. Set `min_recall: 1.01` in `configs/model_config.yaml` and train again.
- The model is registered, but it is **not** made champion and **not** exported. The exit code is 1, which is what fails CI.
- Check the exit code with `echo $LASTEXITCODE`, then change the value back.

**Rollback a model, registry-side.** Run this after training twice, so that 2 versions exist:
```powershell
python -c "from mlflow import MlflowClient; import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); c=MlflowClient(); c.set_registered_model_alias('cancer-classifier','champion','1'); print(c.get_model_version_by_alias('cancer-classifier','champion').version)"
```

## 3. API (no Docker)

```powershell
uvicorn src.api.main:app --reload            # http://localhost:8000/docs  (try it in the browser)
curl.exe -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d "@examples/malignant.json"
curl.exe -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d "@examples/benign.json"
curl.exe -s localhost:8000/model/info
```
**Break it:**
- Send an invalid input: `curl.exe -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{\"radius_mean\": -1}'`. You get a `422` that lists every problem.
- Remove the model and watch startup fail. Rename `artifacts\model` to `artifacts\model_x`, then start uvicorn. It **refuses to start** with a clear error. Rename the folder back.

## 4. Tests and quality checks
```powershell
pytest --cov
black --check src tests scripts; flake8 src tests scripts; mypy src scripts
```
**Break it.** Move the scaler out of the pipeline. Delete the `("scaler", StandardScaler()),` line for `logistic_regression` in `src/models/build.py` and run `pytest`. The leakage-guard test fails. Undo the change.

---

## 5. Docker (start Docker Desktop first)

```powershell
docker build -f docker/Dockerfile -t cancer-api:1.0.0 .
docker images cancer-api
docker run -d --name api -p 8000:8000 cancer-api:1.0.0
docker ps                              # STATUS shows (health: starting) then (healthy)
docker logs -f api                     # JSON logs; Ctrl+C to stop following
docker exec -it api sh                 # a shell INSIDE the container. Try: whoami ; ls /app ; exit
curl.exe -s localhost:8000/ready
```
Things to notice:
- `whoami` prints `app` or `10001`, not root.
- Inside the container there is no `tests/`, `data/` or `venv/`, because `.dockerignore` kept them out.

**Layer caching.** Edit a comment in `src/api/main.py` and rebuild. The `pip install` step says `CACHED`, so the rebuild takes seconds.

**Bad vs good image:**
```powershell
docker build -f docker/Dockerfile.bad -t cancer-api:bad .
docker images cancer-api               # compare SIZE
docker run --rm cancer-api:bad whoami  # root
```

**Break it:**
- Kill the container with `docker kill api; docker ps -a`. Its status shows Exited. With `--restart unless-stopped` (as in Compose), Docker would restart it.

Clean up: `docker rm -f api`

## 6. Docker Compose: the 4-service stack

> **If image downloads fail** with `failed to copy: httpReadSeeker ... EOF`, Docker Hub's download
> servers are dropping your connection. Retrying usually works, because finished layers are cached. If it keeps failing,
> pull the same official images from other registries and give them the Docker Hub names Compose expects:
> ```powershell
> docker pull quay.io/prometheus/prometheus:v3.5.0
> docker tag  quay.io/prometheus/prometheus:v3.5.0 prom/prometheus:v3.5.0
> docker pull mirror.gcr.io/grafana/grafana:12.1.0
> docker tag  mirror.gcr.io/grafana/grafana:12.1.0 grafana/grafana:12.1.0
> ```
> `docker tag` only adds a second name to an image you already have. Nothing is copied.
> For kind's node image, which is also on Docker Hub, simply retry `docker pull` until it completes.

```powershell
docker compose up -d --build
docker compose ps                      # all 4 services; api shows (healthy)
```
| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 (see Status â†’ Targets, then Alerts) |
| Grafana | http://localhost:3000, admin/admin; open Dashboards â†’ Cancer MLOps |

Log training runs into the Compose MLflow server instead of the local file:
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"; python -m scripts.train; Remove-Item Env:MLFLOW_TRACKING_URI
```
**Service networking.** Containers reach each other by service name:
```powershell
docker compose exec prometheus wget -qO- http://api:8000/health
```
**Volumes survive restarts:**
```powershell
docker compose down; docker compose up -d     # MLflow runs and Grafana are still there
docker compose down -v                         # -v deletes the volumes too: data gone
```
**Break it.** Stop the API with `docker compose stop api`. After about 30 seconds the `APIDown` alert fires at http://localhost:9090/alerts. Bring it back with `docker compose start api`.

## 7. Monitoring and drift

Generate traffic, then look at Grafana:
```powershell
python -m scripts.load_test --url http://localhost:8000 --concurrency 10 --duration 60
```
Try these PromQL queries at http://localhost:9090:
```
sum by (path, status) (rate(http_requests_total[1m]))
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{path="/predict"}[1m])))
sum by (diagnosis) (increase(predictions_total[5m]))
```
**Drift:**
```powershell
python -m scripts.simulate_drift --url http://localhost:8000
```
- **Phase 1** (normal traffic) gives `dataset_drift=False`.
- **Phase 2** (sizes Ã—1.4) gives `dataset_drift=True`, with radius, perimeter and area flagged.
- The Grafana drift panel turns red, and `DataDriftDetected` fires in Prometheus.

---

## 8. Kubernetes with kind

**One-time install:** `winget install Kubernetes.kind`, then open a new terminal.

> Use **`127.0.0.1:30080`, not `localhost:30080`.** On Windows, `localhost` tries IPv6 first, but kind's
> port mapping only listens on IPv4. Every new connection then waits for the IPv6 attempt to fail:
> about 0.2 s with curl and about 2 s with Python. We measured it: 220 ms vs 7 ms per request.
```powershell
.\infrastructure\kubernetes\setup-kind.ps1
kubectl -n cancer-mlops get all
kubectl -n cancer-mlops get pods -o wide        # see the pods spread across both worker nodes
curl.exe -s 127.0.0.1:30080/ready
curl.exe -s -X POST 127.0.0.1:30080/predict -H "X-API-Key: local-demo-key" -H "Content-Type: application/json" -d "@examples/benign.json"
```
Watch everything live in a **second terminal**: `kubectl -n cancer-mlops get pods -w`

**Self-healing.** Delete a pod and watch the ReplicaSet replace it:
```powershell
$pod = kubectl -n cancer-mlops get pods -o name | Select-Object -First 1
kubectl -n cancer-mlops delete $pod
kubectl -n cancer-mlops get events --sort-by=.lastTimestamp | Select-Object -Last 10
```

**Manual scaling.** The HPA may scale the deployment back to its own decision:
```powershell
kubectl -n cancer-mlops scale deployment cancer-api --replicas=5
```

**Load balancing.** Count which pod served each request:
```powershell
python -m scripts.load_test --url http://127.0.0.1:30080 --api-key local-demo-key --concurrency 6 --duration 20 --new-connections
python -m scripts.load_test --url http://127.0.0.1:30080 --api-key local-demo-key --concurrency 6 --duration 20
```
- With `--new-connections`, requests spread across all pods.
- Without it, each thread's kept-alive connection sticks to one pod. A Service balances **connections**, not individual requests.

**Autoscaling (HPA).** Generate load and watch the replica count grow:
```powershell
kubectl -n cancer-mlops get hpa -w             # in the second terminal: TARGETS shows cpu% / 60%
python -m scripts.load_test --url http://127.0.0.1:30080 --api-key local-demo-key --concurrency 40 --duration 180 --new-connections
```
- Replicas grow from 3 toward 10.
- After the load stops and 60 seconds of low CPU pass, they shrink back to 3.
- `kubectl top pods -n cancer-mlops` shows the CPU per pod.

**Rolling update and rollback:**
```powershell
docker build -f docker/Dockerfile -t cancer-api:1.0.1 .
kind load docker-image cancer-api:1.0.1 --name cancer-mlops
kubectl -n cancer-mlops set image deployment/cancer-api api=cancer-api:1.0.1
kubectl -n cancer-mlops rollout status deployment/cancer-api
kubectl -n cancer-mlops rollout history deployment/cancer-api
kubectl -n cancer-mlops rollout undo deployment/cancer-api          # back to 1.0.0
```

**Break it: a bad release.** Point the deployment at an image that doesn't exist:
```powershell
kubectl -n cancer-mlops set image deployment/cancer-api api=cancer-api:does-not-exist
kubectl -n cancer-mlops get pods                # 1 new pod in ErrImagePull, the 3 old ones still Running
kubectl -n cancer-mlops describe pod <the-broken-pod>   # read the Events section at the bottom
kubectl -n cancer-mlops rollout undo deployment/cancer-api
```
`maxUnavailable: 0` meant the old pods kept serving the whole time, so users saw no outage.

**Break it: a broken readiness probe.** Point the probe at a path that doesn't exist:
```powershell
kubectl -n cancer-mlops patch deployment cancer-api --type=json -p '[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/readinessProbe/httpGet/path\",\"value\":\"/nope\"}]'
kubectl -n cancer-mlops get pods                # new pod Running but READY 0/1, so it gets no traffic and the rollout stalls
kubectl -n cancer-mlops rollout undo deployment/cancer-api
```

**Break it: out of memory.** The app uses about 125Mi; give it a 60Mi limit. The request must be lowered too:
Kubernetes rejects a request (256Mi) that is bigger than its limit, before anything is deployed.
```powershell
kubectl -n cancer-mlops set resources deployment cancer-api --requests=memory=50Mi --limits=memory=60Mi
kubectl -n cancer-mlops get pods -w             # new pod: OOMKilled (exit code 137) -> CrashLoopBackOff; old pods keep serving
kubectl apply -k infrastructure/kubernetes      # restore the declared state from the YAML files
```

> **`rollout undo` is relative: it means "go back one revision".** If the change you wanted to undo
> never created a revision (for example, Kubernetes rejected it), `undo` takes you back to whatever
> came *before*, which may be a broken version. Safer options:
> `kubectl rollout undo --to-revision=N` (check `kubectl rollout history` first), or re-apply the
> YAML from git with `kubectl apply -k infrastructure/kubernetes`. The files in git are the source of truth.

**Delete everything:** `kind delete cluster --name cancer-mlops`

---

## 9. CI/CD on GitHub

1. Create an empty GitHub repo and push this project (`git add . ; git commit ; git push`).
2. Open the **Actions** tab. `CI` runs on every push.
3. To try CD:
   - Push a tag: `git tag v1.0.0 ; git push --tags`.
   - The image is published to `ghcr.io/<you>/cancer-api`.
   - The deploy job skips unless the `KUBECONFIG_B64` secret exists.
4. To run training: Actions â†’ **Scheduled model training** â†’ **Run workflow**.
