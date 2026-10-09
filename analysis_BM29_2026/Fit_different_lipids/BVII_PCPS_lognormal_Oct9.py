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


PCPS = [
    "BII_S11", #: "15 mM DOPC/DOPS (8/2)",

    "BIX_S15", #: "15 mM DOPC/DOPS (8/2) + 6 μM aSynWT",
    "BIX_S16", #: "15 mM DOPC/DOPS  (8/2) + 9 μM aSynWT",
    "BIX_S17", #: "15 mM DOPC/DOPS (8/2) + 15 μM aSynWT",
    "BIX_S18", #: "15 mM DOPC/DOPS (8/2) + 25 μM aSynWT",
    "BIX_S19", #: "15 mM DOPC/DOPS (8/2) + 50 μM aSynWT",

    # "BVII_S4", #: "15 mM DOPC/DOPS (8/2) + 75 μM aSynWT", # bgr sub removed too much of the curve
    "BVII_S5", #: "15 mM DOPC/DOPS (8/2) + 100 μM aSynWT",
    "BVII_S10", #: "15 mM DOPC/DOPS (8/2) + 75 μM aSynE61V",
    "BVII_S15", #: "15 mM DOPC/DOPS (8/2) + 75 μM aSynA124T",
    "BVII_S20", #: "15 mM DOPC/DOPS (8/2) + 75 μM aSynE46K",

    # "BVII_S6", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynWT",
    # "BVII_S11", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynE61V",
    # "BVII_S16", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynA124T",
    # "BVII_S21", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynE46K",
]

who = PCPS

if __name__ == "__main__":
    # =============================================================================
    DAY = "Oct9"
    task_name = "approaching_DOPC_DOPS"

    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()

    fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym_lognormal(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
    fit_function_name = "ULV_spheric_asym_C4_lognormal"

    for sample in who:
        count = 0
        while (count < 10):
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
            lower_bounds = [   9,          0,      0.8,      0.8,        0.13,      0.13,      0.5,    1.5,       0,    0] 
            upper_bounds = [  25,          1,      2.5,      2.5,         0.6,       0.6,      1.2,      2,       1,    1] 
            p0 = np.random.uniform(lower_bounds, upper_bounds)

            print(f"Now fitting (count={count}) with {fit_function_name} sample {sample} ...")            
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
            target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name / sample
            if not target_folder.exists():
                target_folder.mkdir()

            print("Fit successful with chi2 = ", chi2)
            # print(info_text_print)
            with open(target_folder/f"{sample}_{DAY}.txt", "a", encoding="utf-8") as k:
                k.write(f"\n\n{BM29.sample_dict[sample]} fitted (count={count}) using {fit_function_name} with chi2 = {chi2}\n" + fit_lib_entry + "\n###########################################\n")

            chi2_reference = BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq")
            if chi2 < chi2_reference + 30:
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

            count += 1

    # =======================================================================================

    # ========================== ULV Lognormal + Added Layer Fit ===================================

    fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c4: fit.ULV_lognormal_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, 0, 0, c4)
    fit_function_name = "ULV_lognormal_sph_asym_prot_layer"

    who = [
        "BVII_S6", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynWT",
        "BVII_S11", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynE61V",
        "BVII_S16", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynA124T",
        "BVII_S21", #: "3 mM DOPC/DOPS (8/2) + 100 μM aSynE46K",
    ]

    for sample in who:
        this_file = Path(__file__).resolve()
        analysis_BM29_July2026 = this_file.parent.parent
        target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name / sample
        if not target_folder.exists():
            target_folder.mkdir()
            
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

        count = 0
        while (count < 5):
            # ========================== Setting Bounds ==============================
            # p0         = [        R,  sigma_R,       rho_h1, rho_h2,   sigma_h1, sigma_h2,   sigma_c,      z_h,  z_h3, sigma_h3,   rho_h3,    scale, c4] 
            lower_bounds = [        9,        0,          0.8,    0.8,       0.13,     0.13,       0.6,      1.7,   1.8,      0.5,      0.0,        0,  0] 
            upper_bounds = [       30,      1.5,          2.5,    2.5,        0.6,      0.6,       1.2,      1.9,     6,      1.5,      2.0,   0.0001,  0.01] 
            p0 = np.random.uniform(lower_bounds, upper_bounds)

            print(f"Now fitting (count={count}) with {fit_function_name} sample {sample} ...")            
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
                count += 1
                continue

            param_names = [ "R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit",
                            "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit", "z_h3_fit", "sigma_h3_fit", "rho_h3_fit", "scale_fit", "c4_fit"]
            perr = np.sqrt(np.diag(pcov))
            fit_lib_entry = "["
            for name, value in zip(param_names, popt):
                globals()[name] = value
                fit_lib_entry += f"{value:.4g}, "
            fit_lib_entry = fit_lib_entry[:-2]
            fit_lib_entry += "]"
            print(fit_lib_entry)
            chi2, info_text_print = fit.all_info(q, I, Ierr, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

            print("Fit successful with chi2 = ", chi2)
            print(info_text_print)
            with open(target_folder/f"{sample}_3mM_{DAY}.txt", "a", encoding="utf-8") as k:
                k.write(f"\n\n{BM29.sample_dict[sample]} fitted using {fit_function_name} with chi2 = {chi2}\n" + fit_lib_entry + "\n###########################################\n")

            # chi2_reference = BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq", fit_function=fit_function_name)
            if chi2 < 1e+5:
                #=========================== Plot ===========================================
                q_fit = np.linspace(min(q), max(q), 1000)
                plt.close("all")
                plt.figure(figsize=(12, 7.5))
                plt.errorbar(q, I, Ierr, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
                plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
                plt.loglog(q_fit, fit_function(q_fit, *p0), color='orange', label='initial guess')
                plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, c4_fit), color='black', label='pow law bgr')
                plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, p0[-1]), color='black', label='initial bgr', ls="--")

                plt.xlabel('$q [\\mathrm{nm}^{-1}]$')
                plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
                plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name}')
                plt.grid(True)
                legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
                legend.set_title(info_text_print)
                plt.tight_layout()
                plt.savefig(target_folder / f"{sample}_{fit_function_name}_{count}_{DAY}.png",dpi=80, bbox_inches='tight', pad_inches=0.05)

            count += 1


