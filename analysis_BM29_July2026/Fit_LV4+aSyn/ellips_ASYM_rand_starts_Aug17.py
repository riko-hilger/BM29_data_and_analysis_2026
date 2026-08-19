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

# =============== ELLIPSOIDAL ASYM NO BGR ======================================
# =========================================================================
fit_function_name = "ellipsoid_asym"
fit_function = lambda q,  R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale: fit.ULV_ellipsoid_asym(q, R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale,   0, 0, 0)
DAY = "Aug19"

# -------------- LOAD DATA --------------------------------------------------
sample = "BII_S9" # "BII_S9_15mM_LV4_PBS_FVII"
block = sample.split("_")[0]
block_path = BM29.blockpath(block)
dir_names = [x.name for x in block_path.iterdir()]
for dir_name in dir_names:
    if dir_name[:-13].endswith(sample):
        sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
        q, I, Ierr = np.loadtxt(sub_dat_path, unpack=True)

q     = q
I     = I
I_err_Poisson = Ierr
I_err_op = np.abs(I * 1/30) 
mask = (q <= 0.1) & (q >= 4.1)
I_err_op[mask] = np.abs(I[mask] * 1/15) 
I_err_Poisson[mask] = I_err_Poisson[mask] * 2

thinning = 4
q_thinned = q[::thinning]
I_thinned = I[::thinning]
I_err_thinned = I_err_Poisson[::thinning]

# ========================== Parameters ==========================
R       = 16.13
sigma_R = 4.99
eps  = 1
sigma_eps = 0.437

rho_h1   = 2.8
rho_h2   = 2.5
sigma_h1 = 0.13
sigma_h2 = 0.12
sigma_c = 0.558
z_h = 1.8468

scale = 3.66e-7
"""
BII_S9_15mM_LV4_PBS_FVII using ellipsoid_sym
R = 15.545 (11, 16)
sigma_R = 5.1886 (4, 6)
eps = 0.81488 (0.8, 0.9)
sigma_eps = 0.40363 (0.0, 0.7)
rho_h = 1.3832 (1.0, 2.5)
sigma_h = 0.17433 (0.13, 0.5)
sigma_c = 0.64154 (0.5, 0.9)
z_h = 1.5327 (1.5, 2.2)
scale = 0.00015445 (0, 0.001)
chi2_initial = 8.9667e+08
 
R_fit = 16 ± 0.38935
sigma_R_fit = 5.1402 ± 0.33214
eps_fit = 0.9 ± 0.06058
sigma_eps_fit = 0.46436 ± 0.017759
rho_h_fit = 2.5 ± 0.4217
sigma_h_fit = 0.13619 ± 0.028892
sigma_c_fit = 0.56261 ± 0.032527
z_h_fit = 1.897 ± 0.018051
scale_fit = 4.1576e-07 ± 3.8304e-08
chi2 = 25.125
[16, 5.14, 0.9, 0.4644, 2.5, 0.1362, 0.5626, 1.897, 4.158e-07]
"""

# ========================== Setting Bounds ==============================
# p0           = [        R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale] # ,   c2,   c3,       c4]
# lower_bounds = [       10,        2,     0.8,         0,     1.0,    1.0,       0.15,     0.15,      0.5,      1.5,        0] # ,    0,    0,  -np.inf]
# upper_bounds = [       30,        9,     1.5,       0.6,     2.5,    2.5,        0.6,      0.6,        1,        2,    0.001] # ,  0.2,  0.5,  +np.inf]

p0           = [        R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale] # ,   c2,   c3,       c4]
lower_bounds = [       15,      4.5,    0.82,         0,     2.0,    2.0,       0.10,     0.10,      0.5,      1.8,        0] # ,    0,    0,  -np.inf]
upper_bounds = [       17,      6.0,     1.1,       0.6,       3,      3,        0.5,      0.5,      0.9,     1.95,     1e-5] # ,  0.2,  0.5,  +np.inf]

chi2 = 1e6
chi2_goal = 6
i = 1

if __name__ == "__main__":
    while chi2 > chi2_goal:
        print(f"Iteration {i}, current chi2 = {chi2:.4g}")
        print("Current params: ", p0)

        #======================== Fit ===========================================
        popt, pcov = curve_fit(fit_function, q_thinned, I_thinned, sigma=I_err_thinned, maxfev=10000, p0=p0, bounds = (lower_bounds, upper_bounds))

        param_names = ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h1_fit", "rho_h2_fit",
                  "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit","scale_fit"]

        perr = np.sqrt(np.diag(pcov))
        fit_lib_entry = "["
        for name, value in zip(param_names, popt):
            globals()[name] = value
            fit_lib_entry += f"{value:.4g}, "
        fit_lib_entry = fit_lib_entry[:-2]
        fit_lib_entry += "]"
        print(fit_lib_entry)
        chi2, info_text_print = fit.all_info(q, I, I_err_Poisson, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

        this_file = Path(__file__).resolve()
        analysis_BM29_July2026 = this_file.parent.parent
        target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / f"fit_{fit_function_name}"
        if not target_folder.exists():
            target_folder.mkdir()

        print("Fit successful with chi2 = ", chi2)
        print(info_text_print)
        with open(target_folder/f"info_texts_{DAY}.txt", "a", encoding="utf-8") as k:
            k.write(f"\n\n{BM29.sample_dict[sample]} using {fit_function_name}\n" + info_text_print + "\n" + fit_lib_entry + "\n###########################################\n")

        #=========================== Plot ===========================================
        q_fit = np.linspace(min(q), max(q), 1000)
        plt.close("all")
        plt.figure(figsize=(12, 7.5))
        plt.errorbar(q, I, I_err_Poisson, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
        plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
        plt.loglog(q_fit, fit_function(q_fit, R, sigma_R, eps, sigma_eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale), color='orange', label='initial guess')

        plt.xlabel('$q [1/\\mathrm{nm}]$')
        plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
        plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name} (improv. Poisson err)')
        plt.grid(True)
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
        legend.set_title(info_text_print)
        plt.tight_layout()
        plt.savefig(target_folder / f"{sample}_{fit_function_name}_{i}_{DAY}.png",dpi=400, bbox_inches='tight', pad_inches=0.05)
        fig = fit.EDP_plot(R_fit, sigma_h1_fit, sigma_h2_fit, sigma_c_fit, rho_h1_fit, rho_h2_fit, z_h_fit, z_h_fit, info_text_print, 
                           f'{sample} using {fit_function_name}, nr. {i}')
        fig.savefig(target_folder / f"EDP_{sample}_{fit_function_name}_{i}_{DAY}.png", dpi=400, bbox_inches='tight', pad_inches=0.05)

        p0 = np.random.uniform(lower_bounds, upper_bounds)
        i += 1
