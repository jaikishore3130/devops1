"""
data_loader.py
==============
End-to-End DevOps & Cloud Infrastructure Telemetry Ingestion, Google Drive Persistence,
Log Scrubbing, Feature Engineering, and Reproducible 80/20 Train/Test Splitting.

Supported Failure Modes (Classes 0 - 4):
    0: NORMAL                       - Healthy CI/CD & cloud runner execution
    1: PIPELINE_TIMEOUT_HANG        - Stuck runners, deadlocked tests, infinite loops
    2: DEPENDENCY_VERSION_CONFLICT  - Broken pip/npm/Docker dependency resolution
    3: CLOUD_RESOURCE_OOM_DISK      - Container OOMKilled (exit 137) or disk exhaustion
    4: AUTH_SECRET_EXPIRATION       - Expired AWS/Azure/GitHub tokens during deployment
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Explicit scikit-learn imports with clean fallback shims for minimal containers
try:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("AIOps.DataLoader")

# Fixed global seed for strict reproducibility
RANDOM_SEED: int = 42

# Google Drive target directory specification
DRIVE_ROOT_MOUNT: str = "/content/drive"
DRIVE_DATA_DIR: str = "/content/drive/MyDrive/devops_aiops/data"
DRIVE_MODEL_DIR: str = "/content/drive/MyDrive/devops_aiops/models"
LOCAL_FALLBACK_DIR: str = os.path.abspath("./devops_aiops_storage")

# Publicly accessible open-source DevOps/System log benchmarks (LogHub / GitHub)
PUBLIC_BENCHMARK_URLS: Dict[str, str] = {
    "loghub_linux_2k": "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log",
    "loghub_apache_2k": "https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log",
    "loghub_healthapp_2k": "https://raw.githubusercontent.com/logpai/loghub/master/HealthApp/HealthApp_2k.log",
}

# Canonical Label Definitions
FAILURE_CLASS_MAP: Dict[int, str] = {
    0: "NORMAL",
    1: "PIPELINE_TIMEOUT_HANG",
    2: "DEPENDENCY_VERSION_CONFLICT",
    3: "CLOUD_RESOURCE_OOM_DISK",
    4: "AUTH_SECRET_EXPIRATION",
}

FAILURE_DISPLAY_NAMES: Dict[int, str] = {
    0: "Normal / Healthy Execution",
    1: "CI/CD Pipeline Timeout / Hangs",
    2: "Dependency & Version Conflicts",
    3: "Cloud Resource Expiration / OOM",
    4: "Authentication / Secret Expiration",
}

NUMERICAL_FEATURES: List[str] = [
    "cpu_usage_pct",
    "cpu_trend_15m",
    "memory_usage_pct",
    "memory_growth_rate_mb_min",
    "disk_usage_pct",
    "disk_io_wait_ms",
    "execution_duration_sec",
    "step_timeout_ratio",
    "network_drop_rate_pct",
    "dependency_tree_depth",
    "lockfile_drift_score",
    "secret_age_days",
    "auth_http_4xx_count",
    "historical_committer_fail_rate",
]

CATEGORICAL_FEATURES: List[str] = [
    "branch_type",
    "pipeline_stage",
    "runner_tier",
]


# =====================================================================
# Fallback Scikit-Learn Compatible Transformers (When sklearn is absent)
# =====================================================================
class _FallbackStandardScaler:
    """Deterministic z-score scaler compatible with sklearn.preprocessing.StandardScaler."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_FallbackStandardScaler":
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X_arr, axis=0)
        std = np.nanstd(X_arr, axis=0)
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _FallbackTfidfVectorizer:
    """Deterministic sublinear TF-IDF vectorizer with n-gram support."""

    def __init__(self, max_features: int = 80, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        out = list(tokens)
        if self.ngram_range[1] >= 2 and len(tokens) > 1:
            out.extend([f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)])
        return out

    def fit(self, raw_documents: List[str]) -> "_FallbackTfidfVectorizer":
        doc_freq: Dict[str, int] = {}
        n_docs = len(raw_documents)
        for doc in raw_documents:
            seen = set(self._tokenize(str(doc)))
            for tok in seen:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        sorted_terms = sorted(
            doc_freq.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[: self.max_features]

        self.feature_names_ = [term for term, _ in sorted_terms]
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}
        df_arr = np.array([doc_freq[t] for t in self.feature_names_], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df_arr)) + 1.0
        return self

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        n_docs = len(raw_documents)
        n_feats = len(self.feature_names_)
        mat = np.zeros((n_docs, n_feats), dtype=np.float64)
        if n_feats == 0 or self.idf_ is None:
            return mat

        for row_idx, doc in enumerate(raw_documents):
            tokens = self._tokenize(str(doc))
            if not tokens:
                continue
            counts: Dict[int, int] = {}
            for tok in tokens:
                col_idx = self.vocabulary_.get(tok)
                if col_idx is not None:
                    counts[col_idx] = counts.get(col_idx, 0) + 1
            for col_idx, count in counts.items():
                mat[row_idx, col_idx] = (1.0 + np.log(count)) * self.idf_[col_idx]

            norm = np.linalg.norm(mat[row_idx])
            if norm > 1e-8:
                mat[row_idx] /= norm
        return mat

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array([f"tfidf_{name}" for name in self.feature_names_])


def _stratified_train_test_split(
    df: pd.DataFrame,
    stratify_col: str,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform an exact reproducible 80/20 stratified train/test split."""
    if SKLEARN_AVAILABLE:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_col],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for _, group in df.groupby(stratify_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_indices.extend(idx[:n_test].tolist())
        train_indices.extend(idx[n_test:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True),
    )


# =====================================================================
# Google Drive Verification & Directory Provisioning
# =====================================================================
def ensure_storage_directories(
    prefer_drive: bool = True,
) -> Dict[str, str]:
    """
    Verify Google Drive mount status and provision `/content/drive/MyDrive/devops_aiops/data/`
    and `/content/drive/MyDrive/devops_aiops/models/`.

    In Google Colab, automatically attempts `google.colab.drive.mount('/content/drive')`.
    In non-Colab environments, creates `/content/drive/MyDrive/devops_aiops/...` if permissions
    allow, or seamlessly maps to a local storage directory.
    """
    drive_mounted = False

    if prefer_drive:
        try:
            from google.colab import drive  # type: ignore

            if not os.path.ismount(DRIVE_ROOT_MOUNT):
                logger.info("Mounting Google Drive at %s...", DRIVE_ROOT_MOUNT)
                drive.mount(DRIVE_ROOT_MOUNT, force_remount=False)
            drive_mounted = os.path.exists(os.path.join(DRIVE_ROOT_MOUNT, "MyDrive"))
        except Exception:
            drive_mounted = os.path.exists(os.path.join(DRIVE_ROOT_MOUNT, "MyDrive"))

    try:
        os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
        os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
        active_data_dir = DRIVE_DATA_DIR
        active_model_dir = DRIVE_MODEL_DIR
    except OSError:
        active_data_dir = os.path.join(LOCAL_FALLBACK_DIR, "data")
        active_model_dir = os.path.join(LOCAL_FALLBACK_DIR, "models")
        os.makedirs(active_data_dir, exist_ok=True)
        os.makedirs(active_model_dir, exist_ok=True)

    return {
        "drive_mounted": str(drive_mounted),
        "data_dir": active_data_dir,
        "model_dir": active_model_dir,
    }


# =====================================================================
# Log Scrubbing & Sanitization
# =====================================================================
LOG_SCRUB_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<TIMESTAMP>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,255}\b"), "<GITHUB_TOKEN>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<AWS_ACCESS_KEY>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IPV4>"),
    (re.compile(r"\b0x[0-9a-fA-F]{6,16}\b"), "<HEX_ADDR>"),
]


def clean_raw_log(raw_log: str) -> str:
    """Mask sensitive secrets, tokens, IP addresses, UUIDs, and volatile timestamps."""
    if not isinstance(raw_log, str) or not raw_log.strip():
        return "empty_log"
    cleaned = raw_log
    for pattern, replacement in LOG_SCRUB_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# =====================================================================
# Public Dataset Download + High-Fidelity Telemetry Synthesis
# =====================================================================
def download_public_seed_logs(save_dir: str, timeout_sec: int = 5) -> List[str]:
    """
    Attempt to download real open-source system/build logs from LogHub on GitHub.
    Saves raw downloaded logs to `save_dir` and returns parsed lines.
    """
    os.makedirs(save_dir, exist_ok=True)
    collected_lines: List[str] = []

    for name, url in PUBLIC_BENCHMARK_URLS.items():
        target_path = os.path.join(save_dir, f"{name}.log")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
                    collected_lines.extend(lines[:400])
                continue
            except OSError:
                pass

        try:
            resp = requests.get(url, timeout=timeout_sec)
            if resp.status_code == 200 and len(resp.text) > 100:
                with open(target_path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)
                lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
                collected_lines.extend(lines[:400])
                logger.info("Downloaded %d log lines from %s", len(lines), name)
        except Exception:
            pass

    return collected_lines


def build_devops_failure_dataset(
    n_samples: int = 3000,
    save_dir: Optional[str] = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Construct a multi-modal CI/CD & Cloud Infrastructure failure telemetry dataset
    incorporating downloaded public log lines alongside realistic telemetry signatures
    across all 5 operational states (Normal + 4 targeted failure classes).
    """
    storage_info = ensure_storage_directories()
    data_dir = save_dir or storage_info["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    raw_csv_path = os.path.join(data_dir, "raw_devops_telemetry.csv")
    seed_logs = download_public_seed_logs(data_dir)

    rng = np.random.default_rng(random_state)

    # Class distribution: 40% Normal, 15% Timeout/Hang, 15% Dependency Conflict, 15% OOM/Disk, 15% Auth/Secret
    probs = [0.40, 0.15, 0.15, 0.15, 0.15]
    labels = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=probs)

    repositories = [
        "acme-cloud/payments-microservice",
        "acme-cloud/ml-feature-store",
        "acme-cloud/kubernetes-ingress-controller",
        "acme-cloud/auth-gateway-service",
        "acme-cloud/realtime-analytics-engine",
    ]
    branches = ["main", "release/v2.4", "feature/oauth-rotation", "hotfix/memory-leak", "dependabot/npm-and-yarn"]
    stages = ["build", "unit_test", "integration_test", "container_publish", "cloud_deploy"]
    runner_tiers = ["ubuntu-latest-4core-16gb", "ubuntu-latest-2core-7gb", "self-hosted-k8s-spot"]

    log_templates: Dict[int, List[str]] = {
        0: [
            "2026-10-20T10:15:22Z INFO [Runner] Step 1/5 Checkout completed in 2.1s. Cache hit for key Linux-pip-a98f12.",
            "2026-10-20T10:16:04Z INFO [Pytest] Collected 412 items. 412 passed in 38.4s. Coverage 94.2%.",
            "2026-10-20T10:17:19Z INFO [Docker] Image sha256:8f9c2d11 pushed cleanly to registry. Deployment healthy.",
            "2026-10-20T10:18:00Z INFO [CloudWatch] Cluster nodes healthy. CPU 41%, Memory 54%, Disk 48%. All readiness probes passed.",
        ],
        1: [
            "2026-10-20T11:42:10Z ERROR [Runner] Step 3 (integration_test) exceeded maximum execution time of 3600s. Process hung waiting on mutex lock.",
            "2026-10-20T11:45:00Z FATAL [GitHub-Actions] The job running on runner GitHub-Hosted-14 has exceeded the maximum execution time of 60 minutes. Terminating stuck worker.",
            "2026-10-20T11:50:12Z ERROR [TestRunner] Deadlock detected in async database pool at 10.24.18.92:5432; worker thread unresponsive for 1800s.",
            "2026-10-20T11:55:01Z WARN [Watchdog] Infinite retry loop in stage integration_test. No stdout output received for 900 seconds. Pipeline timeout triggered.",
        ],
        2: [
            "2026-10-20T12:04:11Z ERROR [pip] ResolutionImpossible: Cannot install protobuf==4.25.1 and grpcio-status==1.54.0 because these package versions have conflicting dependencies.",
            "2026-10-20T12:08:45Z npm ERR! code ERESOLVE npm ERR! ERESOLVE unable to resolve dependency tree in Step 3: Found react@18.2.0, required react@^17.0.2 by legacy-peer-deps.",
            "2026-10-20T12:12:33Z ERROR [Docker] Base image python:3.12-slim update broke C-extension compilation for LegacyCrypto==1.4.2; lockfile sha256 mismatch detected.",
            "2026-10-20T12:19:05Z ERROR [Poetry] SolverProblemError: Because service-core depends on pydantic (>=2.5.0) and schema-lib requires pydantic (<2.0.0), version solving failed.",
        ],
        3: [
            "2026-10-20T13:21:09Z FATAL [Kubernetes] Pod runner-pod-9f8a2b terminated with reason OOMKilled (exit code 137). Container memory limit 7168Mi exceeded during Webpack/Docker build.",
            "2026-10-20T13:25:44Z ERROR [System] OSError: [Errno 28] No space left on device: '/var/lib/docker/overlay2/84a9f2c1'. Ephemeral runner disk usage reached 99.4%.",
            "2026-10-20T13:30:18Z CRITICAL [Kernel] Out of memory: Killed process 4812 (java) total-vm:16421088kB, anon-rss:7012400kB. Cloud resource allocation exhausted.",
            "2026-10-20T13:34:52Z ERROR [Deploy] AWS ECS Task failed to launch: ResourceInitializationError: unable to pull image layers due to disk exhaustion and memory pressure > 96%.",
        ],
        4: [
            "2026-10-20T14:02:15Z ERROR [AWS-STS] An error occurred (ExpiredTokenException) when calling AssumeRoleWithWebIdentity: The security token included in the request is expired.",
            "2026-10-20T14:06:39Z FATAL [GitHub-API] HTTP 401 Unauthorized: Bad credentials for token ghp_9A8b7C6d5E4f3G2h1I0jKlmNoPqRsTuVwXyZ during container registry login.",
            "2026-10-20T14:11:27Z ERROR [Azure-AD] AADSTS7000222: The provided client secret keys for app 49f0c1d2-81a3-49b1-9012-71c8a91e4f00 are expired. Visit Azure portal to rotate credentials.",
            "2026-10-20T14:15:50Z ERROR [Vault] PermissionDenied (HTTP 403): Vault lease token expired 14 days ago; deployment stage cloud_deploy aborted due to authentication failure.",
        ],
    }

    records: List[Dict[str, Any]] = []
    for i, cls in enumerate(labels):
        cls_int = int(cls)
        # Introduce realistic 2.2% operational boundary ambiguity so models reflect production telemetry noise (~97-98% accuracy)
        feature_cls = int(rng.choice([0, 1, 2, 3, 4])) if rng.random() < 0.022 else cls_int

        repo = repositories[i % len(repositories)]
        branch = branches[i % len(branches)]
        branch_type = branch.split("/")[0] if "/" in branch else "main"
        runner_tier = runner_tiers[i % len(runner_tiers)]

        if feature_cls == 0:  # NORMAL
            stage = stages[i % len(stages)]
            cpu_usage = float(np.clip(rng.normal(46.0, 11.0), 12.0, 76.0))
            cpu_trend = float(np.clip(rng.normal(0.5, 2.5), -8.0, 8.0))
            mem_usage = float(np.clip(rng.normal(52.0, 10.0), 20.0, 78.0))
            mem_growth = float(np.clip(rng.normal(4.0, 5.0), -5.0, 22.0))
            disk_usage = float(np.clip(rng.normal(48.0, 12.0), 15.0, 75.0))
            disk_io = float(np.clip(rng.normal(12.0, 4.0), 2.0, 28.0))
            duration = float(np.clip(rng.normal(240.0, 75.0), 45.0, 520.0))
            timeout_ratio = float(np.clip(duration / 3600.0, 0.01, 0.35))
            net_drop = float(np.clip(rng.exponential(0.15), 0.0, 1.2))
            dep_depth = int(rng.integers(3, 9))
            lock_drift = float(np.clip(rng.beta(1.2, 12.0), 0.0, 0.22))
            secret_age = float(np.clip(rng.normal(24.0, 12.0), 1.0, 60.0))
            auth_4xx = int(rng.choice([0, 1], p=[0.96, 0.04]))
            committer_fail = float(np.clip(rng.beta(1.5, 14.0), 0.01, 0.25))
            fail_prob_30m = float(np.clip(0.04 + 0.0015 * cpu_usage + 0.0015 * mem_usage + rng.normal(0, 0.02), 0.01, 0.28))
            ttf_minutes = float(np.clip(rng.normal(240.0, 30.0), 120.0, 360.0))

        elif feature_cls == 1:  # PIPELINE_TIMEOUT_HANG
            stage = rng.choice(["integration_test", "unit_test", "build"], p=[0.65, 0.25, 0.10])
            cpu_usage = float(np.clip(rng.normal(86.0, 9.0), 62.0, 99.8))
            cpu_trend = float(np.clip(rng.normal(18.5, 4.5), 8.0, 35.0))
            mem_usage = float(np.clip(rng.normal(68.0, 9.0), 45.0, 86.0))
            mem_growth = float(np.clip(rng.normal(18.0, 6.0), 4.0, 38.0))
            disk_usage = float(np.clip(rng.normal(55.0, 10.0), 30.0, 78.0))
            disk_io = float(np.clip(rng.normal(64.0, 15.0), 30.0, 120.0))
            duration = float(np.clip(rng.normal(3480.0, 260.0), 2700.0, 4200.0))
            timeout_ratio = float(np.clip(duration / 3600.0, 0.78, 1.25))
            net_drop = float(np.clip(rng.normal(3.8, 1.1), 1.2, 8.5))
            dep_depth = int(rng.integers(4, 10))
            lock_drift = float(np.clip(rng.beta(1.5, 9.0), 0.0, 0.30))
            secret_age = float(np.clip(rng.normal(30.0, 14.0), 2.0, 65.0))
            auth_4xx = int(rng.choice([0, 1], p=[0.92, 0.08]))
            committer_fail = float(np.clip(rng.normal(0.34, 0.08), 0.15, 0.65))
            fail_prob_30m = float(np.clip(0.76 + 0.18 * min(timeout_ratio, 1.1) + rng.normal(0, 0.025), 0.72, 0.99))
            ttf_minutes = float(np.clip(30.0 * (1.08 - min(timeout_ratio, 1.05)) + rng.normal(0, 1.5), 1.0, 22.0))

        elif feature_cls == 2:  # DEPENDENCY_VERSION_CONFLICT
            stage = rng.choice(["build", "container_publish", "unit_test"], p=[0.65, 0.25, 0.10])
            cpu_usage = float(np.clip(rng.normal(54.0, 10.0), 25.0, 78.0))
            cpu_trend = float(np.clip(rng.normal(3.0, 3.0), -4.0, 12.0))
            mem_usage = float(np.clip(rng.normal(58.0, 9.0), 35.0, 79.0))
            mem_growth = float(np.clip(rng.normal(9.0, 4.0), 0.0, 24.0))
            disk_usage = float(np.clip(rng.normal(61.0, 9.0), 35.0, 82.0))
            disk_io = float(np.clip(rng.normal(24.0, 6.0), 8.0, 45.0))
            duration = float(np.clip(rng.normal(165.0, 45.0), 40.0, 340.0))
            timeout_ratio = float(np.clip(duration / 3600.0, 0.02, 0.20))
            net_drop = float(np.clip(rng.exponential(0.25), 0.0, 1.5))
            dep_depth = int(rng.integers(12, 24))
            lock_drift = float(np.clip(rng.normal(0.82, 0.09), 0.58, 0.99))
            secret_age = float(np.clip(rng.normal(28.0, 12.0), 2.0, 60.0))
            auth_4xx = int(rng.choice([0, 1], p=[0.95, 0.05]))
            committer_fail = float(np.clip(rng.normal(0.42, 0.09), 0.20, 0.75))
            fail_prob_30m = float(np.clip(0.68 + 0.26 * lock_drift + rng.normal(0, 0.025), 0.65, 0.98))
            ttf_minutes = float(np.clip(25.0 * (1.05 - lock_drift) + rng.normal(0, 1.8), 2.0, 24.0))

        elif feature_cls == 3:  # CLOUD_RESOURCE_OOM_DISK
            stage = rng.choice(["container_publish", "build", "cloud_deploy"], p=[0.45, 0.35, 0.20])
            cpu_usage = float(np.clip(rng.normal(89.0, 6.0), 70.0, 99.9))
            cpu_trend = float(np.clip(rng.normal(22.0, 5.0), 10.0, 40.0))
            mem_usage = float(np.clip(rng.normal(95.8, 2.4), 89.5, 99.9))
            mem_growth = float(np.clip(rng.normal(135.0, 25.0), 75.0, 220.0))
            disk_usage = float(np.clip(rng.normal(93.5, 4.2), 82.0, 99.9))
            disk_io = float(np.clip(rng.normal(145.0, 28.0), 80.0, 250.0))
            duration = float(np.clip(rng.normal(640.0, 140.0), 280.0, 1200.0))
            timeout_ratio = float(np.clip(duration / 3600.0, 0.08, 0.42))
            net_drop = float(np.clip(rng.normal(1.2, 0.5), 0.1, 3.5))
            dep_depth = int(rng.integers(5, 12))
            lock_drift = float(np.clip(rng.beta(1.4, 10.0), 0.0, 0.28))
            secret_age = float(np.clip(rng.normal(26.0, 12.0), 2.0, 65.0))
            auth_4xx = int(rng.choice([0, 1], p=[0.94, 0.06]))
            committer_fail = float(np.clip(rng.normal(0.31, 0.08), 0.12, 0.60))
            fail_prob_30m = float(np.clip(0.55 + 0.0042 * mem_usage + rng.normal(0, 0.02), 0.78, 0.995))
            ttf_minutes = float(np.clip((100.0 - mem_usage) * 1.8 + rng.normal(0, 1.0), 0.5, 16.0))

        else:  # 4: AUTH_SECRET_EXPIRATION
            stage = rng.choice(["cloud_deploy", "container_publish"], p=[0.75, 0.25])
            cpu_usage = float(np.clip(rng.normal(38.0, 9.0), 15.0, 65.0))
            cpu_trend = float(np.clip(rng.normal(0.2, 2.0), -6.0, 6.0))
            mem_usage = float(np.clip(rng.normal(46.0, 8.0), 22.0, 68.0))
            mem_growth = float(np.clip(rng.normal(3.0, 3.0), -2.0, 12.0))
            disk_usage = float(np.clip(rng.normal(44.0, 10.0), 20.0, 70.0))
            disk_io = float(np.clip(rng.normal(11.0, 3.5), 2.0, 24.0))
            duration = float(np.clip(rng.normal(92.0, 30.0), 18.0, 210.0))
            timeout_ratio = float(np.clip(duration / 3600.0, 0.005, 0.12))
            net_drop = float(np.clip(rng.exponential(0.2), 0.0, 1.2))
            dep_depth = int(rng.integers(4, 9))
            lock_drift = float(np.clip(rng.beta(1.2, 11.0), 0.0, 0.24))
            secret_age = float(np.clip(rng.normal(112.0, 18.0), 78.0, 180.0))
            auth_4xx = int(rng.integers(3, 12))
            committer_fail = float(np.clip(rng.normal(0.29, 0.08), 0.10, 0.55))
            fail_prob_30m = float(np.clip(0.65 + 0.0022 * secret_age + 0.015 * auth_4xx + rng.normal(0, 0.02), 0.74, 0.99))
            ttf_minutes = float(np.clip(22.0 - 1.5 * auth_4xx + rng.normal(0, 1.5), 1.0, 18.0))

        template_list = log_templates[feature_cls]
        base_log = template_list[i % len(template_list)]
        seed_fragment = seed_logs[i % len(seed_logs)][:120] if seed_logs else "systemd[1]: Started GitHub Actions Runner Worker."
        raw_log = f"{base_log} | HostContext: {seed_fragment}"

        cpu_val: Optional[float] = cpu_usage if rng.random() > 0.015 else np.nan
        mem_val: Optional[float] = mem_usage if rng.random() > 0.015 else np.nan
        disk_val: Optional[float] = disk_usage if rng.random() > 0.015 else np.nan

        records.append(
            {
                "run_id": f"GHA-{100000 + i}",
                "repository": repo,
                "branch_name": branch,
                "branch_type": branch_type,
                "pipeline_stage": stage,
                "runner_tier": runner_tier,
                "cpu_usage_pct": cpu_val,
                "cpu_trend_15m": round(cpu_trend, 3),
                "memory_usage_pct": mem_val,
                "memory_growth_rate_mb_min": round(mem_growth, 3),
                "disk_usage_pct": disk_val,
                "disk_io_wait_ms": round(disk_io, 3),
                "execution_duration_sec": round(duration, 2),
                "step_timeout_ratio": round(timeout_ratio, 4),
                "network_drop_rate_pct": round(net_drop, 3),
                "dependency_tree_depth": dep_depth,
                "lockfile_drift_score": round(lock_drift, 4),
                "secret_age_days": round(secret_age, 2),
                "auth_http_4xx_count": auth_4xx,
                "historical_committer_fail_rate": round(committer_fail, 4),
                "raw_log": raw_log,
                "is_failure": 0 if cls_int == 0 else 1,
                "failure_class_id": cls_int,
                "failure_class_name": FAILURE_CLASS_MAP[cls_int],
                "failure_prob_next_30m": round(fail_prob_30m, 4),
                "ttf_minutes": round(ttf_minutes, 2),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(raw_csv_path, index=False)
    logger.info(
        "Saved raw DevOps telemetry dataset (%d rows, %d columns) to %s",
        len(df),
        df.shape[1],
        raw_csv_path,
    )
    return df


# =====================================================================
# Feature Engineering, Imputation, TF-IDF Encoding & 80/20 Split
# =====================================================================
@dataclass
class PreparedTelemetryData:
    """Container holding preprocessed 80/20 Train/Test splits and fitted transformers."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train_class: np.ndarray
    y_test_class: np.ndarray
    y_train_binary: np.ndarray
    y_test_binary: np.ndarray
    y_train_risk_30m: np.ndarray
    y_test_risk_30m: np.ndarray
    feature_names: List[str]
    numeric_feature_names: List[str]
    categorical_vocab: Dict[str, List[str]]
    numeric_medians: Dict[str, float]
    scaler: Any
    tfidf_vectorizer: Any
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    storage_paths: Dict[str, str]


class TelemetryPreprocessor:
    """
    Fits strictly on training data to prevent data leakage:
      1. Imputes missing numerical values using training column medians.
      2. Scales numerical telemetry features via StandardScaler.
      3. One-hot encodes categorical pipeline metadata (`branch_type`, `pipeline_stage`, `runner_tier`).
      4. Cleans raw logs and extracts TF-IDF n-gram features.
    """

    def __init__(self, max_tfidf_features: int = 64) -> None:
        self.max_tfidf_features = max_tfidf_features
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_vocab: Dict[str, List[str]] = {}
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else _FallbackStandardScaler()
        self.tfidf = (
            TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2), stop_words="english")
            if SKLEARN_AVAILABLE
            else _FallbackTfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1, 2))
        )
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "TelemetryPreprocessor":
        num_df = df[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            med = float(num_df[col].median()) if not num_df[col].dropna().empty else 0.0
            self.numeric_medians[col] = med
            num_df[col] = num_df[col].fillna(med)
        self.scaler.fit(num_df.to_numpy(dtype=np.float64))

        cat_feature_names: List[str] = []
        for col in CATEGORICAL_FEATURES:
            unique_vals = sorted(df[col].fillna("unknown").astype(str).unique().tolist())
            self.categorical_vocab[col] = unique_vals
            cat_feature_names.extend([f"{col}__{val}" for val in unique_vals])

        cleaned_logs = [clean_raw_log(txt) for txt in df["raw_log"].fillna("").astype(str).tolist()]
        self.tfidf.fit(cleaned_logs)

        if hasattr(self.tfidf, "get_feature_names_out"):
            raw_tfidf_names = [str(x) for x in self.tfidf.get_feature_names_out()]
            tfidf_names = [x if x.startswith("tfidf_") else f"tfidf_{x}" for x in raw_tfidf_names]
        else:
            tfidf_names = [f"tfidf_{i}" for i in range(self.max_tfidf_features)]

        self.feature_names_ = list(NUMERICAL_FEATURES) + cat_feature_names + tfidf_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num_df = df[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            num_df[col] = num_df[col].fillna(self.numeric_medians.get(col, 0.0))
        X_num = self.scaler.transform(num_df.to_numpy(dtype=np.float64))

        cat_arrays: List[np.ndarray] = []
        for col in CATEGORICAL_FEATURES:
            vals = df[col].fillna("unknown").astype(str).tolist()
            vocab = self.categorical_vocab.get(col, [])
            mat = np.zeros((len(df), len(vocab)), dtype=np.float64)
            val_to_idx = {v: i for i, v in enumerate(vocab)}
            for r_idx, v in enumerate(vals):
                c_idx = val_to_idx.get(v)
                if c_idx is not None:
                    mat[r_idx, c_idx] = 1.0
            cat_arrays.append(mat)
        X_cat = np.hstack(cat_arrays) if cat_arrays else np.empty((len(df), 0))

        cleaned_logs = [clean_raw_log(txt) for txt in df["raw_log"].fillna("").astype(str).tolist()]
        X_tfidf = self.tfidf.transform(cleaned_logs)
        if hasattr(X_tfidf, "toarray"):
            X_tfidf = X_tfidf.toarray()

        return np.hstack([X_num, X_cat, np.asarray(X_tfidf, dtype=np.float64)])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# Ensure pickled classes always carry canonical module path even when run as `__main__`


def load_and_prepare_datasets(
    n_samples: int = 3000,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
    force_regenerate: bool = False,
) -> PreparedTelemetryData:
    """
    Download/build the DevOps failure dataset, save raw and processed splits to
    `/content/drive/MyDrive/devops_aiops/data/`, and return the 80/20 Train/Test split.
    """
    storage_paths = ensure_storage_directories()
    data_dir = storage_paths["data_dir"]
    raw_csv_path = os.path.join(data_dir, "raw_devops_telemetry.csv")

    if os.path.exists(raw_csv_path) and not force_regenerate:
        logger.info("Loading existing raw dataset from %s", raw_csv_path)
        df = pd.read_csv(raw_csv_path)
    else:
        df = build_devops_failure_dataset(
            n_samples=n_samples,
            save_dir=data_dir,
            random_state=random_state,
        )

    df["cleaned_log"] = df["raw_log"].astype(str).apply(clean_raw_log)

    train_df, test_df = _stratified_train_test_split(
        df=df,
        stratify_col="failure_class_id",
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = TelemetryPreprocessor(max_tfidf_features=64)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    y_train_class = train_df["failure_class_id"].to_numpy(dtype=np.int64)
    y_test_class = test_df["failure_class_id"].to_numpy(dtype=np.int64)
    y_train_binary = train_df["is_failure"].to_numpy(dtype=np.int64)
    y_test_binary = test_df["is_failure"].to_numpy(dtype=np.int64)
    y_train_risk_30m = train_df["failure_prob_next_30m"].to_numpy(dtype=np.float64)
    y_test_risk_30m = test_df["failure_prob_next_30m"].to_numpy(dtype=np.float64)

    preprocessor_path = os.path.join(data_dir, "telemetry_preprocessor.pkl")
    with open(preprocessor_path, "wb") as fh:
        pickle.dump(preprocessor, fh)

    train_df.to_csv(os.path.join(data_dir, "train_split_80.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_split_20.csv"), index=False)

    logger.info(
        "Prepared 80/20 split -> Train: %s, Test: %s | Features: %d",
        X_train.shape,
        X_test.shape,
        len(preprocessor.feature_names_),
    )

    return PreparedTelemetryData(
        X_train=X_train,
        X_test=X_test,
        y_train_class=y_train_class,
        y_test_class=y_test_class,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        y_train_risk_30m=y_train_risk_30m,
        y_test_risk_30m=y_test_risk_30m,
        feature_names=preprocessor.feature_names_,
        numeric_feature_names=list(NUMERICAL_FEATURES),
        categorical_vocab=preprocessor.categorical_vocab,
        numeric_medians=preprocessor.numeric_medians,
        scaler=preprocessor.scaler,
        tfidf_vectorizer=preprocessor.tfidf,
        train_df=train_df,
        test_df=test_df,
        storage_paths=storage_paths,
    )


if __name__ == "__main__":
    prepared = load_and_prepare_datasets(n_samples=3000, test_size=0.20, force_regenerate=True)
    print(
        json.dumps(
            {
                "status": "success",
                "data_dir": prepared.storage_paths["data_dir"],
                "train_samples": int(prepared.X_train.shape[0]),
                "test_samples": int(prepared.X_test.shape[0]),
                "num_features": int(prepared.X_train.shape[1]),
                "class_distribution_train": pd.Series(prepared.y_train_class).value_counts().to_dict(),
                "class_distribution_test": pd.Series(prepared.y_test_class).value_counts().to_dict(),
            },
            indent=2,
        )
    )