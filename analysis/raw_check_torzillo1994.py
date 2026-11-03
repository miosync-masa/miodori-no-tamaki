"""EOS2 against the two digitized raw PI curves of Torzillo & Vonshak (1994), Fig. 4
(SI Section S4.3). Run from anywhere: python3 analysis/raw_check_torzillo1994.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ph10_fit import ph10_model, ph10_gross
warnings.simplefilter('ignore', OptimizeWarning)
M, C = 0.8135, -1.3552   # scaling law (dataset units)
csv = os.path.join(HERE, '..', 'raw_data', 'data', 'torzillo1994_fig4_PI_curve_control_inhibited.csv')
df = pd.read_csv(csv, comment='#')
I = df['PFD_umol_m2_s'].values.astype(float)
for col in ['control_net_O2', 'photoinhibited_net_O2']:
    P = df[col].values.astype(float)
    R0 = -P[0]; Pm0 = P.max() + R0; a0 = Pm0 / 100
    best = None
    for S in [1, 3, 10, 30, 100]:                     # multi-start on beta
        try:
            popt, _ = curve_fit(ph10_model, I, P, p0=[a0, a0 / S, Pm0, R0], maxfev=20000)
        except RuntimeError:
            continue
        res = ((ph10_model(I, *popt) - P) ** 2).sum()
        if best is None or res < best[1]:
            best = (popt, res)
    a, b, Pm, R = best[0]
    ss = ((P - P.mean()) ** 2).sum()
    bref = 10 ** (M * np.log10(a) + C)
    pred2 = ph10_gross(I, a, bref, Pm) - R             # EOS2: beta discarded
    nrmse = lambda p: np.sqrt(((p - P) ** 2).mean()) / (P.max() - P.min()) * 100
    print(f"{col}: alpha={a:.3f} beta={b:.4f} Pmax={Pm:.0f} R={R:.1f} S={a/b:.1f} "
          f"SAI={np.log10(b)-(M*np.log10(a)+C):+.2f} | Ph10 fit R2={1-best[1]/ss:.3f} | "
          f"EOS2 vs raw: R2={1-((pred2-P)**2).sum()/ss:.3f} NRMSE={nrmse(pred2):.1f}%")
