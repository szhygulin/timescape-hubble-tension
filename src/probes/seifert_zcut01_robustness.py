"""Robustness check: recompute z>=0.01 LCDM -2lnL minimum on a FINE local Om grid (warm-started
from the reported nuisances) to confirm the marginal dBIC=-0.53 sign survives grid refinement."""
import os, json, numpy as np, importlib.util
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec=importlib.util.spec_from_file_location("seifert",
    os.path.join(_ROOT, "src", "probes", "seifert.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
import fit_timescape as F

d=json.load(open(os.path.join(_ROOT, "probes_out", "seifert.json")))['results']['z>=0.010']
L=d['lcdm']; nm=[L['alpha'],L['beta'],np.log(L['Vx']),np.log(L['Vc']),np.log(L['VM'])]  # warm start
ts_min=d['timescape']['neg2lnL']
lcdm_coarse=d['lcdm']['neg2lnL']
case=S.build_case(0.01); print(f"z>=0.01 N={case['n']}; coarse LCDM min={lcdm_coarse:.4f}, TS min={ts_min:.4f}, coarse dBIC={ts_min-lcdm_coarse:+.4f}",flush=True)
xw=np.array(nm)
grid=np.round(np.arange(0.41,0.491,0.01),3)
n2=[]
for om in grid:
    mu=5*np.log10((1+case['zHEL'])*F.D_shape_LCDM(case['zCMB'],om))
    f,xw=S.profile_point(case,mu,xw); n2.append(f)
    print(f"  Om={om:.3f}: -2lnL={f:.4f}",flush=True)
n2=np.array(n2)
bo,bmin,i=S.refine_min(grid,n2)
print(f"\nFINE LCDM min: Om={bo:.4f}  -2lnL={bmin:.4f}  (vs coarse {lcdm_coarse:.4f}, delta={bmin-lcdm_coarse:+.4f})",flush=True)
print(f"FINE dBIC(TS-LCDM) = {ts_min-bmin:+.4f}  ({'TIMESCAPE' if ts_min-bmin<0 else 'LCDM'})",flush=True)
print(f"=> sign {'SURVIVES (still <=0)' if ts_min-bmin<=0.2 else 'CHANGES'} the z>=0.01 timescape/neutral lean",flush=True)
