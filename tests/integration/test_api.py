"""API integration tests: the real FastAPI app, real model bundle, real HTTP requests (in-process).

Protects against: broken endpoints, validation holes, auth bypass, missing metrics,
and a server that starts without a model.
"""

import pytest
from fastapi.testclient import TestClient


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "test"


def test_predict_malignant(client, malignant_case):
    r = client.post("/predict", json=malignant_case)
    assert r.status_code == 200
    body = r.json()
    assert body["diagnosis"] == "malignant"
    assert 0 <= body["malignant_probability"] <= 1
    assert body["model_version"] == "test"


def test_predict_benign(client, benign_case):
    assert client.post("/predict", json=benign_case).json()["diagnosis"] == "benign"


def test_batch_predict_keeps_order(client, malignant_case, benign_case):
    r = client.post("/predict/batch", json={"instances": [malignant_case, benign_case]})
    assert r.status_code == 200
    assert [p["diagnosis"] for p in r.json()["predictions"]] == ["malignant", "benign"]
    assert r.json()["count"] == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(radius_mean=-1.0),  # out of range
        lambda d: d.pop("area_mean"),  # missing field
        lambda d: d.update(radius_mean="big"),  # wrong type
        lambda d: d.update(radius_mea=1.0),  # unknown field (typo)
    ],
    ids=["negative", "missing", "wrong-type", "extra-field"],
)
def test_invalid_input_rejected_with_422(client, malignant_case, mutate):
    payload = dict(malignant_case)
    mutate(payload)
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_limits(client, benign_case):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    assert (
        client.post("/predict/batch", json={"instances": [benign_case] * 1001}).status_code == 422
    )


def test_api_key_enforced_when_configured(bundle_dir, monkeypatch, benign_case):
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    monkeypatch.setenv("API_KEY", "s3cret")
    from src.api.main import app

    with TestClient(app) as c:
        assert c.post("/predict", json=benign_case).status_code == 401
        assert (
            c.post("/predict", json=benign_case, headers={"X-API-Key": "wrong"}).status_code == 401
        )
        assert (
            c.post("/predict", json=benign_case, headers={"X-API-Key": "s3cret"}).status_code == 200
        )


def test_metrics_endpoint_exposes_counters(client, benign_case):
    client.post("/predict", json=benign_case)
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "http_request_duration_seconds_bucket" in text
    assert 'predictions_total{diagnosis="benign"}' in text


def test_drift_needs_enough_samples_then_reports(client, benign_case):
    assert client.get("/drift").json()["status"] == "insufficient_data"
    client.post("/predict/batch", json={"instances": [benign_case] * 40})
    body = client.get("/drift").json()
    assert body["status"] == "ok"
    # 40 copies of one patient look nothing like the training distribution -> drift.
    assert body["dataset_drift"] is True


def test_server_refuses_to_start_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    from src.api.main import app

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
