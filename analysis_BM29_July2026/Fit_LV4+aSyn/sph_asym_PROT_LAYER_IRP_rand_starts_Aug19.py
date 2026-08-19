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

# =============== SPHERIC ASYM PROT LAYER NO BGR ======================================
# =========================================================================
fit_function_name = "ULV_sph_asym_prot_layer"
fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale: fit.ULV_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, 0, 0, 0)
DAY = "Aug19"

# -------------- LOAD DATA --------------------------------------------------
sample = "BIV_S9" # "BII_S9_15mM_LV4_PBS_FVII"
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

# ========================== Parameters ==========================
R = 10
sigma_R = 5.5  # BEST FROM SPHERICAL ASYM FIT
rho_h1 = 1.8
rho_h2 = 1.15
sigma_h1 = 0.3
sigma_h2 = 0.44
sigma_c = 0.5
z_h = 1.76
z_h3 = 2.9
sigma_h3 = 0.3
rho_h3 = 0.5
scale = 5e-7
# ========================== Setting Bounds ==============================
p0           = [        R,  sigma_R,       rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,  z_h3, sigma_h3,   rho_h3,    scale] # ,   c2,   c3,       c4]
lower_bounds = [       14,        7,          0.8,    1.0,        0.4,      0.1,      0.7,      1.7,     3,      0.7,      0.0,        0] #,    0,    0,  -np.inf]
upper_bounds = [       15,        8,          1.0,    2.5,        0.6,      0.3,      1.2,      1.9,     5,      1.2,      2.0,   0.0001] # ,  0.2,  0.5,  +np.inf]


chi2 = 1e6
chi2_goal = 5
i = 16
if __name__ == "__main__":
    while chi2 > chi2_goal:
        print(f"Iteration {i}, current chi2 = {chi2:.4g}")
        print("Current params: ", p0)
        p0 = np.random.uniform(lower_bounds, upper_bounds)

        #======================== Fit ===========================================
        popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_Poisson, maxfev=10000, p0=p0, bounds = (lower_bounds, upper_bounds))

        param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit",
                    "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit", "z_h3_fit", "sigma_h3_fit", "rho_h3_fit", "scale_fit"]

        perr = np.sqrt(np.diag(pcov))
        fit_lib_entry = "["
        for name, value in zip(param_names, popt):
            globals()[name] = value
            fit_lib_entry += f"{value:.4g}, "
        fit_lib_entry = fit_lib_entry[:-2]
        fit_lib_entry += "]"
        # print(fit_lib_entry)
        chi2, info_text_print = fit.all_info(q, I, I_err_Poisson, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

        if chi2 < 14:
            this_file = Path(__file__).resolve()
            analysis_BM29_July2026 = this_file.parent.parent
            target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / f"fit_{fit_function_name}"
            if not target_folder.exists():
                target_folder.mkdir()

            print("Fit successful with chi2 = ", chi2)
            print(info_text_print)
            with open(target_folder/f"info_texts_{DAY}.txt", "a", encoding="utf-8") as k:
                k.write(f"\n\n{BM29.sample_dict[sample]} using {fit_function_name} with IRP fit nr. {i}\n" + info_text_print + "\n" + fit_lib_entry + "\n###########################################\n")

            #=========================== Plot ===========================================
            q_fit = np.linspace(min(q), max(q), 1000)
            plt.close("all")
            plt.figure(figsize=(12, 7.5))
            plt.errorbar(q, I, I_err_Poisson, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
            plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
            plt.loglog(q_fit, fit_function(q_fit, R,  sigma_R,       rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,  z_h3, sigma_h3,   rho_h3,    scale), color='orange', label='initial guess')

            plt.xlabel('$q [1/\\mathrm{nm}]$')
            plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
            plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name} (improv. Poisson error)')
            plt.grid(True)
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
            legend.set_title(info_text_print)
            plt.tight_layout()
            plt.savefig(target_folder / f"{sample}_{fit_function_name}_IRP{i}_{DAY}.png",dpi=400, bbox_inches='tight', pad_inches=0.05)

            fig=fit.EDP_plot(R_fit, sigma_h1_fit, sigma_h2_fit, sigma_c_fit, rho_h1_fit, rho_h2_fit, z_h_fit, z_h_fit, info_text_print,  f'{BM29.sample_dict[sample]} using {fit_function_name} nr. {i}', z_h3_fit, sigma_h3_fit, rho_h3_fit)
            fig.savefig(target_folder / f"EDP_{sample}_{fit_function_name}_IRP{i}.png", dpi=400, bbox_inches='tight', pad_inches=0.05)

        i += 1
