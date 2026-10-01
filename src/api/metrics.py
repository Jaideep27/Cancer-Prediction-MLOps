"""Prometheus metrics exposed at GET /metrics.

Counter   = a number that only goes up (requests served). Prometheus computes rates from it.
Histogram = counts observations into buckets (latency <=5ms, <=10ms, ...) -> percentiles.
Gauge     = a number that goes up and down (current drift share).
Info      = constant labels (which model version is loaded).

Label values must come from a SMALL fixed set (route templates, status codes),
never from user input, or every unique value creates a new time series.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests served", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("predictions_total", "Predictions made, by outcome", ["diagnosis"])
MALIGNANT_PROBABILITY = Histogram(
    "prediction_malignant_probability",
    "Distribution of predicted P(malignant)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_INFO = Info("model", "Currently loaded model")
DRIFT_SHARE = Gauge("data_drift_share", "Share of features flagged as drifted at last check")
DRIFT_DETECTED = Gauge("data_drift_detected", "1 if dataset-level drift was detected at last check")
DRIFT_WINDOW_SIZE = Gauge("data_drift_window_size", "Recent inputs held for drift checks")
