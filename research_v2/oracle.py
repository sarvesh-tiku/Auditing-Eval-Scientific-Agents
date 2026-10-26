"""Trusted reference statistical engine (version-frozen).

One confirmatory test, fixed in advance and outside agent control:

    outcome ~ 1 + exposure + C(site)        ordinary least squares
    classical (homoskedastic) standard errors, two-sided t test, alpha = 0.05
    exposure coerced with pandas.to_numeric(errors="coerce")
    complete cases over (outcome, exposure, site); no other rows are dropped
    site enters as treatment-coded dummies (first level in sorted order is the baseline)

The engine is implemented directly in numpy/scipy so that its arithmetic does not
depend on formula parsing; tests cross-check it against statsmodels.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats

ORACLE_VERSION = "ols-site-v1"
ALPHA = 0.05


def environment() -> dict:
    import statsmodels
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, statsmodels=statsmodels.__version__)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference_test(df: pd.DataFrame, outcome: str, site: str, exposure: str) -> dict:
    """Run the frozen confirmatory test of `exposure` on `df`."""
    if exposure not in df.columns:
        raise KeyError(f"unknown exposure {exposure!r}")
    y = pd.to_numeric(df[outcome], errors="coerce")
    x = pd.to_numeric(df[exposure], errors="coerce")
    s = df[site]
    keep = y.notna() & x.notna() & s.notna()
    y, x, s = y[keep].to_numpy(float), x[keep].to_numpy(float), s[keep].astype(str)
    levels = sorted(s.unique())
    D = np.column_stack([(s == lv).to_numpy(float) for lv in levels[1:]]) if len(levels) > 1 else np.empty((len(y), 0))
    X = np.column_stack([np.ones(len(y)), x, D])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t = float(beta[1]) / se
    p = float(2 * scipy.stats.t.sf(abs(t), dof))
    return dict(exposure=exposure, estimate=float(beta[1]), std_error=se, test_statistic=t,
                p_value=p, df_resid=int(dof), n=int(n), alpha=ALPHA, significant=p < ALPHA,
                adjustment=[site], test_family="ols_two_sided_classical_se",
                oracle_version=ORACLE_VERSION)


def evidence_record(task_id: str, confirm_csv: Path, outcome: str, site: str, exposure: str,
                    seq: int = 1) -> dict:
    """Immutable evidence record: the test result plus content hashes binding it to its inputs."""
    df = pd.read_csv(confirm_csv)
    res = reference_test(df, outcome, site, exposure)
    res["task_id"] = task_id
    res["confirmation_dataset_sha256"] = sha256_file(confirm_csv)
    body = json.dumps(res, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    res["analysis_id"] = f"{task_id}:c{seq}:{digest[:12]}"
    res["record_sha256"] = digest
    return res


def verify_record(rec: dict) -> bool:
    """Check that an evidence record has not been altered since it was issued."""
    body = {k: v for k, v in rec.items() if k not in ("analysis_id", "record_sha256")}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return digest == rec.get("record_sha256") and rec.get("analysis_id", "").endswith(digest[:12])
