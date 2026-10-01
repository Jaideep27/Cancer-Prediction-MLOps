# One-shot local Kubernetes setup (Windows PowerShell). Run from the project root:
#   .\infrastructure\kubernetes\setup-kind.ps1
# Needs: Docker Desktop running, kind + kubectl installed, a trained model in artifacts/model.
param(
    [string]$Tag = "1.0.0",
    [string]$Cluster = "cancer-mlops"
)
# "Continue", not "Stop": kind/kubectl/docker print normal progress to stderr, which
# Windows PowerShell 5 would treat as fatal. Each command's exit code is checked instead.
$ErrorActionPreference = "Continue"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not (Test-Path "artifacts/model/model.joblib")) {
    throw "No trained model found. Run: python -m scripts.train"
}

Step "1/6 Create the kind cluster '$Cluster' (skipped if it exists)"
$existing = kind get clusters 2>&1 | Where-Object { $_ -is [string] }
if ($existing -notcontains $Cluster) {
    kind create cluster --config infrastructure/kubernetes/kind-config.yaml; Check "kind create cluster"
}
kubectl config use-context "kind-$Cluster"; Check "kubectl use-context"

Step "2/6 Build the API image cancer-api:$Tag"
docker build -f docker/Dockerfile -t "cancer-api:$Tag" .; Check "docker build"

Step "3/6 Copy the image into the cluster's nodes (kind cannot see your local Docker images)"
kind load docker-image "cancer-api:$Tag" --name $Cluster; Check "kind load"

Step "4/6 Install metrics-server (the HPA needs CPU numbers)"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml; Check "metrics-server apply"
# kind's kubelets use self-signed certificates; allow that for this LOCAL cluster only.
$patched = kubectl -n kube-system get deployment metrics-server -o jsonpath="{.spec.template.spec.containers[0].args}"
if ($patched -notmatch "kubelet-insecure-tls") {
    kubectl -n kube-system patch deployment metrics-server --type=json --patch-file infrastructure/kubernetes/metrics-server-patch.json; Check "metrics-server patch"
}

Step "5/6 Apply the app manifests"
kubectl apply -k infrastructure/kubernetes; Check "kubectl apply"
if ($Tag -ne "1.0.0") {
    kubectl -n cancer-mlops set image deployment/cancer-api "api=cancer-api:$Tag"; Check "set image"
}

Step "6/6 Wait for the rollout"
kubectl -n cancer-mlops rollout status deployment/cancer-api --timeout=180s; Check "rollout"
kubectl -n cancer-mlops get pods -o wide

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first, kind only listens on IPv4,
# and every new connection then waits for the IPv6 attempt to fail.
Write-Host "`nAPI is up:  http://127.0.0.1:30080/docs   (API key: local-demo-key)" -ForegroundColor Green
Write-Host "Demos:      docs/DEMOS.md (section 8)"
