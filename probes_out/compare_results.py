#!/usr/bin/env python3
"""Compare regenerated results_*.json in the worktree root against the
pre-run snapshot in probes_out/orig_snapshot/. Flags any leaf value with
relative difference > 1e-6 (or any structural/string mismatch).
Also dumps src/scan_results.npz array inventory for the fv-split probe agent.
"""
import json, glob, os, sys
import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(WT, "probes_out", "orig_snapshot")

def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: diffs.append((path + "/" + k, "<missing in orig>", b[k])); continue
            if k not in b: diffs.append((path + "/" + k, a[k], "<missing in new>")); continue
            walk(a[k], b[k], path + "/" + k, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)}", f"len {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b: return
        denom = max(abs(a), abs(b), 1e-300)
        if abs(a - b) / denom > 1e-6:
            diffs.append((path, a, b))
    else:
        if a != b: diffs.append((path, a, b))

print("=== results_*.json: snapshot vs regenerated ===")
any_diff = False
for snap_path in sorted(glob.glob(os.path.join(SNAP, "results*.json"))):
    name = os.path.basename(snap_path)
    # results_rasanen.json lives in src/, the rest in the worktree root
    new_path = os.path.join(WT, "src" if name == "results_rasanen.json" else "", name)
    with open(snap_path) as f: orig = json.load(f)
    with open(new_path) as f: new = json.load(f)
    diffs = []
    walk(orig, new, "", diffs)
    same_bytes = open(snap_path, "rb").read() == open(new_path, "rb").read()
    print(f"\n{name}: byte-identical={same_bytes}  value-diffs>{1e-6:g}rel: {len(diffs)}")
    for p, a, b in diffs:
        any_diff = True
        print(f"  DIFF {p}: orig={a!r}  new={b!r}")
print("\nOVERALL:", "MISMATCHES FOUND" if any_diff else "all values match within 1e-6 relative")

print("\n=== src/scan_results.npz inventory ===")
npz = np.load(os.path.join(WT, "src", "scan_results.npz"))
for k in npz.files:
    a = npz[k]
    print(f"{k}: shape={a.shape} dtype={a.dtype}", end="")
    if a.ndim == 1 and a.size > 1 and np.issubdtype(a.dtype, np.floating):
        step = np.diff(a)
        print(f" min={a.min():.6g} max={a.max():.6g}"
              f" uniform_step={step[0]:.6g}" if np.allclose(step, step[0]) else
              f" min={a.min():.6g} max={a.max():.6g} (non-uniform)", end="")
    elif a.ndim == 0:
        print(f" value={a}", end="")
    print()

def grid_report(gname, cname):
    if gname in npz.files and cname in npz.files:
        g, c = npz[gname], npz[cname]
        i = int(np.argmin(c))
        print(f"{cname} vs {gname}: grid [{g[0]:.6g}, {g[-1]:.6g}] step={g[1]-g[0]:.6g} n={g.size}; "
              f"min chi2={c[i]:.4f} at {gname}={g[i]:.6g} (index {i})")
        # 1-sigma (dchi2<=1) interval on the grid
        lo = g[c <= c[i] + 1.0]
        if lo.size: print(f"  dchi2<=1 interval: [{lo.min():.6g}, {lo.max():.6g}]")

grid_report("fv_grid", "chi2_TS")
grid_report("om_grid", "chi2_LCDM")
for k in npz.files:
    if k not in ("fv_grid", "chi2_TS", "om_grid", "chi2_LCDM"):
        a = npz[k]
        if a.ndim == 0:
            print(f"scalar {k} = {a}")
        elif a.size <= 12:
            print(f"array {k} = {np.array2string(np.asarray(a), precision=6)}")
