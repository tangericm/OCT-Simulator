"""Experiment tracking for the ANCHOR·FIT·JUDGE calibration loop — dependency-free.

Every calibration run is an immutable, reproducible experiment: a run folder holding the
full config, the θ evaluated, the whole metric panel, the DE incumbent trajectory, figures,
and an environment fingerprint, plus one appended row in ``runs/manifest.jsonl``. No MLflow;
just JSON/CSV, so any row re-runs bit-for-bit from its ``git_sha`` + ``seeds_key``.

See ``docs/CALIBRATION_ROUTINE.md`` §5 for the schema this writes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
MANIFEST = os.path.join(RUNS_ROOT, "manifest.jsonl")


def git_sha(short: bool = True) -> str:
    """Short git SHA of the octsim repo, or 'nogit' when unavailable."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if short:
            args = ["git", "rev-parse", "--short", "HEAD"]
        out = subprocess.check_output(
            args, cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def utc_stamp(now=None) -> str:
    """Compact UTC timestamp ``YYYYMMDDTHHMMSSZ``. ``now`` (a struct_time-like) may be
    injected for determinism; defaults to the wall clock."""
    import time
    t = now if now is not None else time.gmtime()
    return time.strftime("%Y%m%dT%H%M%SZ", t)


def env_fingerprint() -> str:
    """A short environment description for ``env.txt`` (reproducibility)."""
    lines = [
        f"git_sha={git_sha(short=False)}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"host={platform.node()}",
    ]
    for mod in ("numpy", "scipy"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}={m.__version__}")
        except Exception:
            lines.append(f"{mod}=?")
    return "\n".join(lines) + "\n"


@dataclass
class ExperimentRun:
    """A single tracked calibration experiment (one run folder + one manifest row)."""

    target_corpus: str
    note: str = ""
    parent_run_id: str | None = None
    runs_root: str = RUNS_ROOT
    run_id: str = field(default="", init=False)
    path: str = field(default="", init=False)

    def start(self, *, stamp: str | None = None) -> "ExperimentRun":
        sha = git_sha()
        ts = stamp or utc_stamp()
        self.run_id = f"{ts}_{self.target_corpus}_{sha}"
        self.path = os.path.join(self.runs_root, self.run_id)
        os.makedirs(os.path.join(self.path, "figures"), exist_ok=True)
        with open(os.path.join(self.path, "env.txt"), "w") as fh:
            fh.write(env_fingerprint())
        return self

    # ------------------------------------------------------------------ artifacts
    def write_json(self, name: str, obj) -> None:
        with open(os.path.join(self.path, name), "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)

    def write_config(self, cfg_dict: dict) -> None:
        self.write_json("config.json", cfg_dict)

    def write_theta(self, theta_dict: dict) -> None:
        self.write_json("theta.json", theta_dict)

    def write_metrics(self, metrics: dict) -> None:
        self.write_json("metrics.json", metrics)

    def write_incumbent(self, rows: list) -> None:
        """rows: list of dicts with keys generation, best_D, wall_s, evals."""
        path = os.path.join(self.path, "incumbent.csv")
        cols = ["generation", "best_D", "wall_s", "evals"]
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    def figure_path(self, name: str) -> str:
        return os.path.join(self.path, "figures", name)

    def append_manifest(self, row: dict) -> None:
        os.makedirs(self.runs_root, exist_ok=True)
        row = dict(row)
        row.setdefault("run_id", self.run_id)
        row.setdefault("target_corpus", self.target_corpus)
        row.setdefault("parent_run_id", self.parent_run_id)
        row.setdefault("note", self.note)
        with open(os.path.join(self.runs_root, "manifest.jsonl"), "a") as fh:
            fh.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:
        pass
    return str(o)


def read_manifest(path: str = MANIFEST) -> list:
    """Load all manifest rows (list of dicts); empty list if none."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows
