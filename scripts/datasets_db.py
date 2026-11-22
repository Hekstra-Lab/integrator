"""SQLite database of the shoebox datasets and the chain each came from: sim run, noise run, integration run.

The registry `docs/datasets.yaml` (committed) holds what no file records;
`build` reads it plus every manifest it names -- `manifest*.json` of a sim
or noise run, the integrated MTZ of an integration run, `dataset.yaml` and
the `counts.npy` header of a dataset, and the tracked configs whose
`data_dir` is a dataset path -- and upserts the lot. Rows are keyed by
name, so a rebuild updates in place and the database can always be
regenerated from the registry and the files; it is never edited by hand.

Tables (foreign keys enforced): sim_runs, noise_runs, integration_runs,
datasets, dataset_configs; view `lineage` joins a dataset to its whole chain.
The file lives on lab storage, which is not purged; rollback journalling,
since SQLite's WAL mode needs shared memory that network filesystems lack.

    python scripts/datasets_db.py build                  # on the cluster, after git pull
    python scripts/datasets_db.py query "SELECT name, n_reflections, integer_valued FROM datasets"
    python scripts/datasets_db.py query "SELECT * FROM lineage WHERE dataset = 'mos010_pc_all_25'"
    python scripts/datasets_db.py schema
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = "/n/lab_storage/hekstra_lab/people/aldama/integrator_data/datasets.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sim_runs (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, dir TEXT NOT NULL, simulator TEXT,
  command TEXT, mos_spread REAL, en_steps INTEGER, mos_doms INTEGER, xtal_size REAL, numimg INTEGER,
  total_deg REAL, nabc TEXT, spec_file TEXT, extra_args TEXT, commits TEXT, manifest TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS noise_runs (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, sim_run_id INTEGER REFERENCES sim_runs(id),
  dir TEXT NOT NULL, tool TEXT, gain REAL, readout REAL, calib REAL, psf_mm REAL, adc REAL, seeds TEXT,
  detector TEXT, tool_version TEXT, manifest TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS integration_runs (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, noise_run_id INTEGER REFERENCES noise_runs(id),
  dir TEXT NOT NULL, software TEXT, version TEXT, prediction TEXT, mtz TEXT,
  n_reflections INTEGER, n_images INTEGER, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, path TEXT UNIQUE NOT NULL,
  kind TEXT CHECK (kind IN ('simulated', 'real')), integration_run_id INTEGER REFERENCES integration_runs(id),
  images TEXT, cut_by TEXT, status TEXT, mode TEXT CHECK (mode IN ('poly', 'mono') OR mode IS NULL),
  n_reflections INTEGER, box_h INTEGER, box_w INTEGER, box_d INTEGER, counts_dtype TEXT,
  integer_valued INTEGER, nonint_fraction REAL, counts_bytes INTEGER, raw_mean REAL, raw_var REAL,
  space_group TEXT, cell_a REAL, cell_b REAL, cell_c REAL, cell_alpha REAL, cell_beta REAL, cell_gamma REAL,
  beam_x REAL, beam_y REAL, lambda_min REAL, lambda_max REAL, physics_source TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS dataset_configs (
  dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE, config TEXT NOT NULL,
  PRIMARY KEY (dataset_id, config));
CREATE VIEW IF NOT EXISTS lineage AS
  SELECT d.name AS dataset, d.kind, d.mode, d.n_reflections, d.box_h || 'x' || d.box_w AS box,
         d.integer_valued, i.name AS integration_run, i.software, i.version, i.prediction,
         n.name AS noise_run, n.tool AS noise_tool, n.gain, n.readout, n.calib, n.psf_mm, n.detector,
         s.name AS sim_run, s.simulator, s.mos_spread, s.en_steps, s.mos_doms, s.xtal_size, s.spec_file, d.path
  FROM datasets d
  LEFT JOIN integration_runs i ON d.integration_run_id = i.id
  LEFT JOIN noise_runs n ON i.noise_run_id = n.id
  LEFT JOIN sim_runs s ON n.sim_run_id = s.id;
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = DELETE")
    con.executescript(SCHEMA)
    return con


def upsert(con, table: str, row: dict) -> int:
    cols = list(row)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
           f"ON CONFLICT(name) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in cols if c != 'name')}")
    con.execute(sql, [row[c] for c in cols])
    return con.execute(f"SELECT id FROM {table} WHERE name = ?", (row["name"],)).fetchone()[0]


def ref(con, table: str, name: str | None):
    if not name:
        return None
    r = con.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchone()
    if r is None:
        raise SystemExit(f"registry refers to {table} '{name}', which is not defined above it")
    return r[0]


def manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def dataset_fields(p: Path, rng) -> dict:
    spec_path = p / "dataset.yaml"
    if not spec_path.exists():
        return dict(status="missing dataset.yaml")
    spec = yaml.safe_load(spec_path.read_text())
    g, cr = spec.get("geometry", {}), spec.get("crystal", {}) or {}
    st = (spec.get("stats", {}) or {}).get("raw") or [None, None]
    counts = p / (spec.get("files", {}) or {}).get("counts", "counts.npy")
    c = np.load(counts, mmap_mode="r")
    if np.issubdtype(c.dtype, np.integer):
        frac = 0.0
    else:
        idx = np.sort(rng.choice(c.shape[0], size=min(2000, c.shape[0]), replace=False))
        v = np.asarray(c[idx], dtype=np.float64)
        frac = float(np.mean(v[np.isfinite(v)] % 1 != 0))
    cell = list(cr.get("cell") or [None] * 6)
    bc = list(cr.get("beam_center_px") or [None, None])
    return dict(
        status="ok", mode="poly" if spec.get("polychromatic") else "mono", n_reflections=spec.get("n_reflections"),
        box_h=g.get("h"), box_w=g.get("w"), box_d=g.get("d"), counts_dtype=str(c.dtype),
        integer_valued=int(frac == 0), nonint_fraction=frac, counts_bytes=counts.stat().st_size,
        raw_mean=st[0], raw_var=st[1], space_group=cr.get("space_group"),
        cell_a=cell[0], cell_b=cell[1], cell_c=cell[2], cell_alpha=cell[3], cell_beta=cell[4], cell_gamma=cell[5],
        beam_x=bc[0], beam_y=bc[1], lambda_min=cr.get("lambda_min"), lambda_max=cr.get("lambda_max"),
        physics_source=(spec.get("physics", {}) or {}).get("source"))


def mtz_counts(path: Path):
    try:
        import reciprocalspaceship as rs
        d = rs.read_mtz(str(path))
        return len(d), int(d["BATCH"].nunique()) if "BATCH" in d else None
    except Exception as e:  # recorded as missing, not fatal: the registry may name a run not yet finished
        print(f"  {path}: {type(e).__name__}: {e}")
        return None, None


def configs_using(path: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-F", path, "--", "configs"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def build(con, registry: Path) -> None:
    reg = yaml.safe_load(registry.read_text())
    now = dt.datetime.now().isoformat(timespec="seconds")
    rng = np.random.default_rng(0)
    for e in reg.get("sim_runs", []):
        d = Path(e["dir"])
        mpath = Path(e["manifest"]) if Path(e["manifest"]).is_absolute() else d / e["manifest"]
        m, prm = manifest(mpath), manifest(mpath).get("params", {})
        upsert(con, "sim_runs", dict(
            name=e["name"], dir=str(d), simulator="datsimx", command=m.get("datsimx_command"),
            mos_spread=prm.get("mos_spread"), en_steps=prm.get("en_steps"), mos_doms=prm.get("mos_doms"),
            xtal_size=prm.get("xtal_size"), numimg=prm.get("numimg"), total_deg=prm.get("total_deg"),
            nabc=prm.get("nabc"), spec_file=prm.get("spec_file"), extra_args=m.get("extra_datsimx_args"),
            commits=json.dumps(m.get("code", {})), manifest=str(mpath), ingested_at=now))
    for e in reg.get("noise_runs", []):
        m = manifest(Path(e["manifest"]))
        nz = m.get("noise", {})
        upsert(con, "noise_runs", dict(
            name=e["name"], sim_run_id=ref(con, "sim_runs", e.get("sim_run")), dir=e["dir"], tool=nz.get("tool"),
            gain=nz.get("gain"), readout=nz.get("readout"), calib=nz.get("calib"), psf_mm=nz.get("psf_mm"),
            adc=nz.get("adc"), seeds=nz.get("seeds"), detector=e.get("detector"),
            tool_version=(m.get("code", {}) or {}).get("nanoBragg_noise"), manifest=e["manifest"], ingested_at=now))
    for e in reg.get("integration_runs", []):
        n_refl, n_img = mtz_counts(Path(e["dir"]) / e["mtz"]) if e.get("mtz") else (None, None)
        upsert(con, "integration_runs", dict(
            name=e["name"], noise_run_id=ref(con, "noise_runs", e.get("noise_run")), dir=e["dir"],
            software=e.get("software"), version=e.get("version"), prediction=e.get("prediction"), mtz=e.get("mtz"),
            n_reflections=n_refl, n_images=n_img, ingested_at=now))
    for e in reg.get("datasets", []):
        row = dict(name=e["name"], path=e["path"], kind=e["kind"],
                   integration_run_id=ref(con, "integration_runs", e.get("integration_run")),
                   images=e.get("images"), cut_by=e.get("cut_by"), ingested_at=now)
        row.update(dataset_fields(Path(e["path"]), rng))
        did = upsert(con, "datasets", row)
        con.execute("DELETE FROM dataset_configs WHERE dataset_id = ?", (did,))
        con.executemany("INSERT INTO dataset_configs VALUES (?, ?)", [(did, c) for c in configs_using(e["path"])])
        print(f"  {e['name']}: {row.get('status')}")
    con.commit()


def show(con, sql: str) -> None:
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = [["" if v is None else str(v) for v in r] for r in cur.fetchall()]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="upsert the registry and every manifest it names")
    b.add_argument("--registry", default=str(ROOT / "docs" / "datasets.yaml"))
    q = sub.add_parser("query", help="run one SQL statement and print the result")
    q.add_argument("sql")
    sub.add_parser("schema", help="print the schema")
    a = ap.parse_args()
    con = connect(a.db)
    if a.cmd == "build":
        build(con, Path(a.registry))
        show(con, "SELECT dataset, kind, mode, n_reflections, box, integer_valued, integration_run, noise_tool, "
                  "gain, sim_run FROM lineage ORDER BY kind DESC, dataset")
    elif a.cmd == "query":
        show(con, a.sql)
    else:
        print(SCHEMA)


if __name__ == "__main__":
    main()
