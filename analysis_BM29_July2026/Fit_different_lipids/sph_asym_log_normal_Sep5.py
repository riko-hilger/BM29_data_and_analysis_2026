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
DAY = "Sep5"
task_name = "lognormal_experiments_BVI"

who = [      
    #"BIX_S2",
    #"BVI_S1",
    #"BVI_S2",
    #"BVI_S3",
    #"BVI_S4",
    #"BVI_S5",
    #"BVI_S6",
    #"BVI_S7",
    #"BVI_S8",
    #"BVI_S9",
    #"BVI_S10",

    "BIX_S1",
    "BVI_S11",
    "BVI_S12",
    "BVI_S13",
    "BVI_S14",
    "BVI_S15",
    "BVI_S16",
    "BVI_S17",
    "BVI_S18",
    "BVI_S19",
    "BVI_S20",
 ]

if __name__ == "__main__":
    for sample in who:
        count = 0
        chi2 = 1e+6
        
        while (count < 2):
            if count == 0:
                p0 = [9.25, 0.4552, 1.261, 1.071, 0.3214, 0.3518, 0.6763, 1.906, 7.089e-07, 0.0005629]
            if count == 1:
                p0 = [10.7, 0.5328, 0.8911, 1.563, 0.3464, 0.305, 0.6757, 1.877, 5.457e-07, 0.0008902]
            print(f"Now fitting with {fit_function_name} sample {sample} ...")
            # -------------- LOAD DATA --------------------------------------------------
            block = sample.split("_")[0]
            block_path = BM29.blockpath(block)
            dir_names = [x.name for x in block_path.iterdir()]
            for dir_name in dir_names:
                if dir_name[:-13].endswith(sample):
                    sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
                    q, I, I_err_Poisson = np.loadtxt(sub_dat_path, unpack=True)

            mask = (q >= 0.07) & (q <= 5.1)
            q    = q[mask]
            I    = I[mask]
            I_err_Poisson = I_err_Poisson[mask]

            Ierr = BM29.Poisson_error_focus_minimum_highq(q, I_err_Poisson)

            thinning = 2
            q_thinned = q[::thinning]
            I_thinned = I[::thinning]
            I_err_thinned = Ierr[::thinning]
            
            # ========================== Setting Bounds ==============================
            # p0         = [   R,    sigma_R,   rho_h1,   rho_h2,    sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,   c4] 
            lower_bounds = [ 9.0,          0,      0.8,      0.8,        0.13,      0.13,      0.5,    1.5,       0,    0] 
            upper_bounds = [  25,          1,      2.5,      2.5,         0.6,       0.6,      1.2,      2,       1,    1] 
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

            param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit", "z_h_fit", "scale_fit", "c4_fit"]

            perr = np.sqrt(np.diag(pcov))
            fit_lib_entry = "["
            for name, value in zip(param_names, popt):
                globals()[name] = value
                fit_lib_entry += f"{value:.4g}, "
            fit_lib_entry = fit_lib_entry[:-2]
            fit_lib_entry += "]"
            print(fit_lib_entry)
            chi2, info_text_print = fit.all_info(q, I, Ierr, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

            this_file = Path(__file__).resolve()
            analysis_BM29_July2026 = this_file.parent.parent
            target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
            if not target_folder.exists():
                target_folder.mkdir()

            print("Fit successful with chi2 = ", chi2)
            print(info_text_print)
            with open(target_folder/f"info_texts_{DAY}.txt", "a", encoding="utf-8") as k:
                k.write(f"\n\n{BM29.sample_dict[sample]} fitted using {fit_function_name} with chi2 = {chi2}\n" + fit_lib_entry + "\n###########################################\n")
    
            #=========================== Plot ===========================================
            q_fit = np.linspace(min(q), max(q), 1000)
            plt.close("all")
            plt.figure(figsize=(12, 7.5))
            plt.errorbar(q, I, Ierr, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
            plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
            plt.loglog(q_fit, fit_function(q_fit, *p0), color='orange', label='initial guess')
            plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, c4_fit), color='black', label='pow law bgr')
            plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, p0[-1]), color='black', label='initial bgr', ls="--")

            plt.xlabel('$q [1/\\mathrm{nm}]$')
            plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
            plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name}')
            plt.grid(True)
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
            legend.set_title(info_text_print)
            plt.tight_layout()
            plt.savefig(target_folder / f"{sample}_{fit_function_name}_{count}_{DAY}.png",dpi=200, bbox_inches='tight', pad_inches=0.05)
            # fig = fit.EDP_plot(R_fit, sigma_h1_fit, sigma_h2_fit, sigma_c_fit, rho_h1_fit, rho_h2_fit, z_h_fit, z_h_fit, info_text_print, 
            #                 f'{sample} using {fit_function_name}')
            # fig.savefig(target_folder / f"EDP_{sample}_{fit_function_name}_{DAY}.png", dpi=200, bbox_inches='tight', pad_inches=0.05)
            # f = plt.figure(figsize=(12, 7.5))
            # from scipy.stats import lognorm
            # sigma = sigma_R_fit
            # R0 = R_fit
            # R = np.linspace(0, R_fit + 50, 1000)
            # P = lognorm.pdf(R, s=sigma, scale=R0)
            # plt.plot(R, P)
            # f.savefig(target_folder / f"Polydisp_{sample}_{fit_function_name}_{count}_{DAY}.png", dpi=200, bbox_inches='tight', pad_inches=0.05)

            # p0 = np.random.uniform(lower_bounds, upper_bounds)
            count += 1
