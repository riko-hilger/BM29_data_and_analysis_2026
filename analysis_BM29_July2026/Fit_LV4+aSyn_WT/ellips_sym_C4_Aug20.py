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

# =============== ELLIPSOIDAL SYM NO BGR ======================================
# =========================================================================
fit_function_name = "ellipsoid_sym_C4"
fit_function = lambda q,  R,  sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale, c4: fit.ULV_ellipsoid(q, R,  sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale,   0, 0, c4)
DAY = "Aug20"

who = [
    "BII_S9",

    "BIV_S1",
    "BIV_S2",
    "BIV_S3",
    "BIV_S4",
    "BIV_S5",
    "BIV_S6",
    "BIV_S7",
    "BIV_S8",
    "BIV_S9",
    "BIV_S10",

    "BIV_S21",
    "BIV_S22",
    "BIV_S23",
    "BIV_S24",
    "BIV_S25",
    "BIV_S26",
    "BIV_S27",
    "BIV_S28",
    "BIV_S29",
    "BIV_S30"
]


if __name__ == "__main__":
    for sample in who:
        print(f"Now fitting with {fit_function_name} sample {sample} ...")
        # -------------- LOAD DATA --------------------------------------------------
        block = sample.split("_")[0]
        block_path = BM29.blockpath(block)
        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
                q, I, I_err_Poisson = np.loadtxt(sub_dat_path, unpack=True)

        q     = q
        I     = I
        I_err_op = BM29.op_error_focus_minimum_high(q, I_err_Poisson)

        thinning = 4
        q_thinned = q[::thinning]
        I_thinned = I[::thinning]
        I_err_thinned = I_err_op[::thinning]

        # ========================== Parameters ==========================
        p0 = [16.13, 4.991, 1, 0.4378, 2.8, 0.1209, 0.5588, 1.876, 3.666e-07, 0]
        # ========================== Setting Bounds ==============================
        # p0         = [        R,  sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale,    c4] 
        lower_bounds = [       10,      4.0,     0.7,       0.0,    1.0,    0.10,        0.5,      1.7,        0,     0] 
        upper_bounds = [       22,        9,     1.2,       0.7,    2.8,     0.5,        0.9,      2.0,   0.0001,     1] 

        #======================== Fit ===========================================
        try:
            popt, pcov = curve_fit(
                fit_function,
                q_thinned,
                I_thinned,
                sigma=I_err_thinned,
                maxfev=5000,
                p0=p0,
                bounds=(lower_bounds, upper_bounds)
            )
        except RuntimeError:
            print("Maximum number of evaluations reached — skipping this fit.")
            continue

        param_names = ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h_fit", 
                    "sigma_h_fit","sigma_c_fit","z_h_fit","scale_fit", "c4_fit"]

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
        plt.loglog(q_fit, fit_function(q_fit, *p0), color='orange', label='initial guess')

        plt.xlabel('$q [1/\\mathrm{nm}]$')
        plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
        plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name} (op. error)')
        plt.grid(True)
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
        legend.set_title(info_text_print)
        plt.tight_layout()
        plt.savefig(target_folder / f"{sample}_{fit_function_name}_{DAY}.png",dpi=400, bbox_inches='tight', pad_inches=0.05)
        # fig = fit.EDP_plot(R_fit, sigma_head, sigma_head, sigma_tail, rho_head, rho_head, eps_head, eps_head, info_text_print, 
        #             f'{sample} using {fit_function_name}, nr. {i}')
        # fig.savefig(target_folder / f"EDP_{sample}_{fit_function_name}_{i}_{DAY}.png", dpi=400, bbox_inches='tight', pad_inches=0.05)

