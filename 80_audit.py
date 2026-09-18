#!/usr/bin/env python3
"""Final audit before this goes to anyone else.

Five questions, each answered against the files rather than from memory.

  A  Does the handoff file reproduce the frozen number on its own? Someone
     who receives only that parquet must be able to recompute the headline.
  B  Do the numbers written into the specification match the numbers in the
     result file they were taken from?
  C  Do the counts and date spans quoted in the inventory match the stores?
  D  Is anything in the repository that should not travel -- credentials, a
     personal name, a company name, or the word this project does not use?
  E  Is the recovered column consistent with the original where both exist?

Anything that fails prints FAIL and the reason.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
fails, warns = [], []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def warn(label, detail=""):
    print(f"  WARN  {label}" + (f"   {detail}" if detail else ""))
    warns.append(label)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name.split("_")[0],
                                                  ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- A
print("\nA. does the handoff file reproduce the frozen number by itself?")
ef = load_module("78_efficiency")
th = load_module("64_threats_flevel")
HARD = th.CTRL + th.LONG
hp = ROOT / "handoff" / "kr_crowding_events.parquet"
lock = json.loads((ROOT / "results" / "lock_final.json").read_text())
ref = lock["variants"]["raw return, 60 day"]

d = pd.read_parquet(hp)
print(f"       file has {len(d):,} rows, {len(d.columns)} columns")
missing = [c for c in ["abn60", "surprise", "i_flow20_v2", "inst_days20"] + HARD
           if c not in d.columns]
check(not missing, "every column the specification names is present",
      f"missing {missing}" if missing else "")

if not missing:
    x, y, S, T = ef.pieces(d, "abn60", False, False)
    r = ef.coef(x, y, S, T, 1e4)
    print(f"       recomputed {r['coef']:+.3f} bp   two-way t {r['t_two_way']:+.3f}"
          f"   n={r['events']:,}  seasons={r['seasons']}")
    print(f"       recorded   {ref['coef']:+.3f} bp   two-way t {ref['t_two_way']:+.3f}"
          f"   n={ref['events']:,}  seasons={ref['seasons']}")
    check(abs(r["coef"] - ref["coef"]) < 0.01, "coefficient reproduces")
    check(abs(r["t_two_way"] - ref["t_two_way"]) < 0.01, "two-way t reproduces")
    check(r["events"] == ref["events"], "event count reproduces")

# ---------------------------------------------------------------- B
print("\nB. do the specification's numbers match the result file?")
spec_path = ROOT / "handoff" / "KR_CROWDING_SPEC.md"
check(spec_path.exists(), "specification file exists")
if spec_path.exists():
    txt = spec_path.read_text()
    ic = lock["variants"]["rank outcome, 60 day"]
    for label, value, pat in (
            ("headline bp", ref["coef"], r"([+-]?\d+\.\d)\s*bp per standard"),
            ("headline t", ref["t_two_way"], r"two-way t\s*([+-]\d+\.\d\d)"),
            ("event count", ref["events"], r"([\d,]+) events over")):
        m = re.search(pat, txt)
        if not m:
            warn(f"could not locate {label} in the specification text")
            continue
        got = float(m.group(1).replace(",", ""))
        check(abs(got - float(value)) < 0.15, f"{label} in the text matches",
              f"text {got} vs result {float(value):.2f}")
    for must in ("inst_days20", "two-way", "placebo", "2020-03-16",
                 "information coefficient"):
        check(must in txt, f"specification mentions {must!r}")

# ---------------------------------------------------------------- C
print("\nC. are the stores fresh, and do the panels hold their shape?")
import datetime as _dt
today = _dt.date.today()
freshness = {"foreign_ownership": 4, "market": 4, "investor_flow": 4}
for store, allow in freshness.items():
    fs = sorted(glob.glob(str(ROOT / "data" / "raw" / store / "**" / "*.parquet"),
                          recursive=True))
    if not fs:
        check(False, f"{store} has files")
        continue
    last = Path(fs[-1]).stem[-8:]
    d_last = _dt.date(int(last[:4]), int(last[4:6]), int(last[6:]))
    lag = (today - d_last).days
    check(lag <= allow, f"{store} is current",
          f"{len(fs):,} files, newest {d_last}, {lag} days behind")

sh = glob.glob(str(ROOT / "data" / "raw" / "shorting" / "**" / "*.parquet"),
               recursive=True)
if len(sh) < 100:
    warn("short-selling series has never been backfilled", f"{len(sh)} files")

n_inv = len(glob.glob(str(ROOT / "data" / "investor" / "*.parquet")))
check(n_inv > 3000, "investor sub-type ticker count", f"{n_inv} on disk")

ev = pd.read_parquet(ROOT / "results" / "event_panel_v6.parquet")
check(len(ev) == 123757, "event panel row count", f"{len(ev):,}")
kinds = ev["kind"].value_counts().to_dict()
check(kinds.get("periodic") == 100296 and kinds.get("provisional") == 23461,
      "event split by kind", str(kinds))
check(str(ev["D"].min().date()) == "2011-03-18"
      and str(ev["D"].max().date()) == "2026-06-01",
      "event date span", f"{ev['D'].min().date()} .. {ev['D'].max().date()}")

# ---------------------------------------------------------------- D
print("\nD. repository hygiene")
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=ROOT).stdout.split()
check(".env" not in tracked, ".env is not tracked")
ALLOWED_DATA = {"data/names_en.csv"}
stray = [t for t in tracked if t.startswith("data/") and t not in ALLOWED_DATA]
check(not stray, "no raw data is tracked", str(stray[:5]))

# only a literal assignment counts; os.getenv, masking and help text do not
secret_pat = re.compile(
    r"(KRX_ID|KRX_PW|API_KEY|PASSWORD|SECRET)\s*=\s*['\"](?!<|your-|\*)\S{6,}",
    re.I)
banned_pat = re.compile(r"darkice|jungan\.park|automat(ic|ed|ion)", re.I)
hits_secret, hits_banned = [], []
for t in tracked:
    p = ROOT / t
    if not p.exists() or p.suffix not in (".py", ".md", ".sh", ".txt", ".yml",
                                          ".yaml", ".json", ".html", ".ipynb"):
        continue
    try:
        body = p.read_text(errors="ignore")
    except Exception:
        continue
    if secret_pat.search(body):
        hits_secret.append(t)
    if banned_pat.search(body):
        hits_banned.append(t)
check(not hits_secret, "no credential assignments in tracked files",
      str(hits_secret[:5]))
generated = {"docs/record.html", "docs/index.html", "docs/event.html",
             "docs/monitor.html"}
# the checker holds the patterns as literals, so it matches itself
hits_banned = [h for h in hits_banned if Path(h).name != Path(__file__).name]
authored = [h for h in hits_banned if h not in generated]
check(not authored, "nothing avoided appears in an authored file",
      str(authored[:8]))
if [h for h in hits_banned if h in generated]:
    warn("a generated page carries the avoided word",
         str([h for h in hits_banned if h in generated]))

st = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                    text=True, cwd=ROOT).stdout.strip().splitlines()
dirty = [l for l in st if not l.startswith("??")]
if dirty:
    warn("tracked files modified but not committed", f"{len(dirty)} files")
    for l in dirty[:8]:
        print(f"        {l}")
else:
    check(True, "no tracked file left uncommitted")

# ---------------------------------------------------------------- E
print("\nE. is the recovered column consistent with the original?")
both = ev["i_flow20"].notna() & ev["i_flow20_v2"].notna()
rho = float(ev.loc[both, "i_flow20"].corr(ev.loc[both, "i_flow20_v2"]))
cov_v1 = float(ev["i_flow20"].notna().mean())
cov_v2 = float(ev["i_flow20_v2"].notna().mean())
print(f"       overlap {int(both.sum()):,}   correlation {rho:.4f}")
print(f"       coverage {cov_v1:.3f} -> {cov_v2:.3f}")
check(rho > 0.97, "recovered column agrees with the original on the overlap")
check(cov_v2 > 0.95, "recovered coverage is what the record claims")
check(int(ev["no_inst20"].fillna(False).sum()) > 0,
      "the no-participation state is carried, not silently zero-filled")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 68)
if fails:
    print(f"{len(fails)} FAILURE(S):")
    for f in fails:
        print(f"  - {f}")
else:
    print("no failures")
if warns:
    print(f"{len(warns)} warning(s): " + "; ".join(warns))
print("=" * 68)
raise SystemExit(1 if fails else 0)
