import numpy as np
from scipy import stats
# q0 LCDM: q0 = 1.5*Om - 1
print("LCDM q0 @ Om=0.30513 (joint, Table II):", round(1.5*0.30513316475074526-1,4))
print("LCDM q0 @ Om=0.30    (BAO+CMB-only, Sec V.A prose):", round(1.5*0.30-1,4))
# w0wa q0 from results_joint: Om=0.31148, w0=-0.87138
Om=0.31148253616297333; w0=-0.8713844069033887; ODE=1-Om
q0 = 0.5*(Om + (1+3*w0)*ODE)
print("w0wa q0 (Table II / Sec V.A):", round(q0,4))
# fresh delta_req
print("fresh delta_req 72.998/61.0-1:", round(72.99788265971678/61.0-1,5))
print("paper ref delta 73.04/61.7-1:", round(73.04/61.7-1,5))
# bootstrap sigma mappings
print("one-sided sigma for p=3/4000:", round(stats.norm.isf(3/4000),3))
print("one-sided sigma for p=0.008:", round(stats.norm.isf(0.008),3))
print("LCDM ref T:", round(5.68/np.sqrt(1.04**2+0.54**2),3))
# Buchert rounding
print("Buchert 1558.5 -> ~1.6e3? 2sf:", float('%.2g'%1558.5))
# chi2/dof LCDM DR2
print("DR2 lcdm chi2/dof 0.891 -> 0.89 / 0.9")
