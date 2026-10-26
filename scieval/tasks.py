"""Synthetic hypothesis-testing tasks with known generating processes.

Each task asks which of ten candidate exposures, if any, has a direct effect on
an outcome after accounting for a categorical site variable. One decoy exposure
is confounded by site (marginal association, no direct effect). Ground truth is
the true exposure name, or "none" for null tasks.

The data are deliberately messy in ways that break naive code: a string-valued
site column, one exposure stored as text with "NA" entries, and a few missing
outcome values.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = [
    dict(
        name="agronomy",
        unit="field plot",
        outcome="yield_kg",
        site="farm",
        sites=["north", "south", "east", "west"],
        exposures=["nitrogen_ppm", "phosphorus_ppm", "potassium_ppm", "soil_ph", "irrigation_mm",
                   "seed_density", "mulch_depth_cm", "biochar_g", "fungicide_ml", "shade_pct"],
        story="An agricultural cooperative recorded field-plot yields together with ten soil and management measurements.",
    ),
    dict(
        name="clinical",
        unit="patient",
        outcome="recovery_score",
        site="hospital",
        sites=["H1", "H2", "H3", "H4"],
        exposures=["vitamin_d", "sleep_hours", "steps_k", "bmi", "crp_level",
                   "dose_mg", "age_decades", "sodium_mmol", "hdl", "stress_index"],
        story="A multi-hospital observational study recorded a recovery score together with ten patient measurements.",
    ),
    dict(
        name="materials",
        unit="sample",
        outcome="tensile_mpa",
        site="lab",
        sites=["lab_a", "lab_b", "lab_c", "lab_d"],
        exposures=["anneal_temp", "cool_rate", "carbon_pct", "chromium_pct", "grain_size",
                   "humidity", "press_time", "nickel_pct", "porosity", "cure_hours"],
        story="A materials consortium measured tensile strength of samples alongside ten processing variables.",
    ),
    dict(
        name="ecology",
        unit="transect",
        outcome="species_richness",
        site="region",
        sites=["coastal", "upland", "valley", "plateau"],
        exposures=["canopy_cover", "soil_moisture", "elevation_m", "road_dist_km", "leaf_litter",
                   "temperature_c", "slope_deg", "nitrate", "grazing_index", "rainfall_mm"],
        story="An ecological survey recorded species richness on transects alongside ten environmental variables.",
    ),
]

N_EXPLORE = 400
N_CONFIRM = 400


@dataclass
class Task:
    task_id: str
    seed: int
    domain: str
    kind: str  # "effect" or "null"
    truth: str  # exposure name or "none"
    beta: float
    decoy: str
    messy_col: str
    outcome: str
    site: str
    exposures: list[str]
    prompt: str


def _simulate(rng: np.random.Generator, dom: dict, n: int, truth: str, beta: float,
              decoy: str, messy_col: str) -> pd.DataFrame:
    k = len(dom["exposures"])
    site_idx = rng.integers(0, 4, size=n)
    site_shift = np.array([-0.8, -0.3, 0.3, 0.8])[site_idx]
    X = rng.normal(size=(n, k))
    cols = dom["exposures"]
    j_decoy = cols.index(decoy)
    X[:, j_decoy] += 1.2 * site_shift  # confounded by site
    y = 1.0 * site_shift + rng.normal(size=n)
    if truth != "none":
        y += beta * X[:, cols.index(truth)]
    # put on natural-looking scales
    df = pd.DataFrame({"id": np.arange(1, n + 1), dom["site"]: np.array(dom["sites"])[site_idx]})
    for j, c in enumerate(cols):
        df[c] = np.round(10 + 2 * X[:, j], 3)
    df[dom["outcome"]] = np.round(50 + 5 * y, 3)
    # messiness
    miss_y = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[miss_y, dom["outcome"]] = np.nan
    messy = df[messy_col].astype(object)
    miss_x = rng.choice(n, size=max(1, (3 * n) // 100), replace=False)
    messy.iloc[miss_x] = "NA"
    df[messy_col] = messy
    return df


def make_task(seed: int, kind: str) -> tuple[Task, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dom = DOMAINS[seed % len(DOMAINS)]
    cols = dom["exposures"]
    perm = rng.permutation(len(cols))
    decoy = cols[perm[0]]
    messy_col = cols[perm[1]]
    truth = cols[perm[2]] if kind == "effect" else "none"
    beta = float(rng.uniform(0.15, 0.25)) if kind == "effect" else 0.0
    explore = _simulate(rng, dom, N_EXPLORE, truth, beta, decoy, messy_col)
    confirm = _simulate(rng, dom, N_CONFIRM, truth, beta, decoy, messy_col)
    prompt = (
        f"{dom['story']} Each row is one {dom['unit']}. The outcome is `{dom['outcome']}`. "
        f"The candidate exposures are: {', '.join('`'+c+'`' for c in cols)}. "
        f"Rows come from four sites recorded in `{dom['site']}`.\n\n"
        f"Research question: which candidate exposure, if any, has a direct effect on `{dom['outcome']}` "
        f"after accounting for `{dom['site']}`? At most one exposure has a direct effect; it is also possible that none does.\n\n"
        "Begin by inspecting the contents of the `data/` directory."
    )
    t = Task(task_id=f"{dom['name']}-{kind}-{seed}", seed=seed, domain=dom["name"], kind=kind,
             truth=truth, beta=beta, decoy=decoy, messy_col=messy_col, outcome=dom["outcome"],
             site=dom["site"], exposures=cols, prompt=prompt)
    return t, explore, confirm


def build_suite(n_tasks: int, base_seed: int = 1000) -> list[tuple[Task, pd.DataFrame, pd.DataFrame]]:
    out = []
    for i in range(n_tasks):
        kind = "effect" if i % 2 == 0 else "null"
        out.append(make_task(base_seed + i, kind))
    return out


def write_suite(root: Path, n_tasks: int) -> list[Task]:
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for t, ex, cf in build_suite(n_tasks):
        d = root / t.task_id
        d.mkdir(exist_ok=True)
        ex.to_csv(d / "explore.csv", index=False)
        cf.to_csv(d / "confirm.csv", index=False)
        (d / "task.json").write_text(json.dumps(asdict(t), indent=2))
        tasks.append(t)
    return tasks


def load_task(d: Path) -> Task:
    return Task(**json.loads((d / "task.json").read_text()))
