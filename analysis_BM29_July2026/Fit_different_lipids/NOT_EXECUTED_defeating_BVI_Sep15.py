from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import fit_lib as fit
import BM29_lib as BM29
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = False
from scipy import optimize
from scipy.optimize import curve_fit



# =========================================================================
fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym_lognormal(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
fit_function_name = "ULV_spheric_asym_C4_lognormal"
DAY = "Sep15"
task_name = "lognormal_BVI_S17"

who = [
    "BVI_S17",
]




who = [
    "BIX_S1",
    # "BVI_S11",
    # "BVI_S12",
    # "BVI_S13",
    "BVI_S14",
    # "BVI_S15",
    # "BVI_S16",
    # "BVI_S17",
    # "BVI_S18",
    "BVI_S19",
    "BVI_S20",
]