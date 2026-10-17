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

who = [
    # "BXII_S1", #: "BXII_S1_15mM_LV4_30nm",
    "BXII_S3", #: "BXII_S3_15mM_LV4_100nm",
    "BXII_S4", #: "BXII_S4_15mM_LV4_200nm",
    # "BXII_S6", #: "BXII_S6_15mM_DOPC",
]
"""
    1. Spheric asym (BB) ambiguity
    2. Spheric asym lognormal 
    3. Ellipsoid sym (few attempts)
    4. Ellipsoid asym 
"""

if __name__ == "__main__":
    # =============================================================================
    DAY = "Oct15"
    """
    # =======================================================================================
    # 1. Spheric asym (BB) ambiguity
    # =======================================================================================
    task_name = "BXII_pure_sph_asym"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()
    fit_function_name = "ULV_spheric_asym_C4"
    fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
    
    def average_deviation_percent(a, b):
        deviations = [abs(x - y) / abs(x) * 100 for x, y in zip(a, b)]
        return sum(deviations) / len(deviations)
    threshold = 10
    
    for sample in who:
        print(f"Now working for task {task_name} on sample {sample} ...")
        # -------------- LOAD DATA --------------------------------------------------
        block = sample.split("_")[0]
        block_path = BM29.blockpath(block)
        this_file = Path(__file__).resolve()
        analysis_BM29_July2026 = this_file.parent.parent
        target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name / sample
        if not target_folder.exists():
            target_folder.mkdir()

        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
                q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)
        mask = (q >= 0.07) & (q <= 5.1)
        q    = q[mask]
        I    = I[mask]
        Ierr_Poisson = Ierr_Poisson[mask]
        I_err_op = BM29.Poisson_error_focus_minimum_highq(q, Ierr_Poisson)
        fit_list = []
        fits = BM29.best_fits[sample]["ULV_spheric_asym_C4"]
        for key in fits:
            fit_list.append(fits[key])

        count = 0
        chi2 = 1e6
        chi2_threshold = BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq") + 20
        print("Current best fit has chi2 ", BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq"))

        if sample == "BXII_S1":
            R_low_lim_vals = [9, 11, 13.5]
            R_max_lim = 20
            sigma_R_max_lim = 8
        elif sample == "BXII_S3":
            R_low_lim_vals = [10, 15, 20, 40]
            R_max_lim = 65
            sigma_R_max_lim = 35
        elif sample == "BXII_S4":
            R_low_lim_vals = [10, 15, 20, 70]
            R_max_lim = 115
            sigma_R_max_lim = 70
        i = 0
        for R_lowlim in R_low_lim_vals:
            for r_uplim in [1.4, 1.6, 2.5]:
                for s1_uplim in [0.25, 0.15, 0.1]:
                    for s2_lowlim in [0.1, 0.15, 0.25]:
                        for r_lowlim in [0.7, 0.8, 1, 1.2]:
                                count += 1
                                plt.close()
                                # ========================== Setting Bounds ==============================
                                # p0         = [        R,          sigma_R,   rho_h1,   rho_h2,   sigma_h1,   sigma_h2,  sigma_c,    z_h,   scale,      c4]
                                lower_bounds = [  R_lowlim,               1, r_lowlim, r_lowlim,   s1_uplim,  s2_lowlim,      0.5,    1.7,       0,       0]
                                upper_bounds = [ R_max_lim, sigma_R_max_lim,  r_uplim,  r_uplim,        0.6,        0.6,        1,      2,    1e-3,    1e-2]
                                p0 = np.random.uniform(lower_bounds, upper_bounds)
                                
                                #======================== Fit ===========================================
                                try:
                                    popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_op, maxfev=10000, p0=p0, bounds = (lower_bounds, upper_bounds))
                                except (RuntimeError, ValueError) as error:
                                    print(f"Fit attempt {count} failed for {sample}: {error}")
                                    continue
                                            
                                param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit", "z_h_fit", "scale_fit", "c4_fit"]
                                perr = np.sqrt(np.diag(pcov))
                                fit_lib_entry = "["
                                for name, value in zip(param_names, popt):
                                    globals()[name] = value
                                    fit_lib_entry += f"{value:.4g}, "
                                fit_lib_entry = fit_lib_entry[:-2]
                                fit_lib_entry += "]"
                                chi2, info_text_print = fit.all_info(q, I, I_err_op, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

                                if chi2 <= chi2_threshold:
                                    novelty_list = [average_deviation_percent(popt, previous_fit) for previous_fit in fit_list]
                                    if all(novelty > threshold for novelty in novelty_list):
                                        fit_list.append(popt)
                                        print(f"Found successful fit nr. {i} for {sample} at attempt {count} with chi2 = {chi2:.4g}")
                                        print(fit_lib_entry)  
                                        # print("\n")          
                                        with open(target_folder/f"info_texts_{sample}.txt", "a", encoding="utf-8") as k:
                                            k.write(f"\n\n{BM29.sample_dict[sample]} fit nr. {i} with chi2 = {chi2:.4g}\n" + fit_lib_entry + "\n###########################################\n")
                                        plt.figure(figsize=(12, 7.5))
                                        plt.errorbar(q, I, I_err_op, marker="o", markersize=2,markeredgewidth=1,ecolor="Blue", markeredgecolor='Blue', markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
                                        plt.loglog(q, fit_function(q, *popt), color='red', label='Fit')
                                        plt.loglog(q, fit_function(q, *p0), color='orange', label='initial guess')
                                        plt.xlabel('$q [1/\\mathrm{nm}]$')
                                        plt.ylabel('$I(q)$')
                                        plt.title(f'{sample} using spheric_asym + C4, fit nr. {i}')
                                        plt.grid(True)
                                        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
                                        legend.set_title(info_text_print)
                                        plt.tight_layout()
                                        plt.savefig(target_folder / f"{fit_function_name}_{i}.png",dpi=200, bbox_inches='tight', pad_inches=0.05)
                                        i += 1

    # =======================================================================================
    # 2. Spheric asym lognormal 
    # =======================================================================================
    task_name = "BXII_pure_sph_asym_lognormal"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()
    fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym_lognormal(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
    fit_function_name = "ULV_spheric_asym_C4_lognormal"

    for sample in who:
        count = 0
        while (count < 6):
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

            if sample == "BXII_S1":
                R_max_lim = 20
                sigma_R_max_lim = 8
            elif sample == "BXII_S3":
                R_max_lim = 65
                sigma_R_max_lim = 35
            elif sample == "BXII_S4":
                R_max_lim = 115
                sigma_R_max_lim = 70
            
            # ========================== Setting Bounds ==============================
            # p0         = [           R,         sigma_R,   rho_h1,   rho_h2,    sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,   c4] 
            lower_bounds = [          12,               0,      0.8,      0.8,        0.13,      0.13,      0.5,    1.5,       0,    0] 
            upper_bounds = [   R_max_lim, sigma_R_max_lim,      2.5,      2.5,         0.6,       0.6,      1.2,      2,       1,    1] 
            p0 = np.random.uniform(lower_bounds, upper_bounds)

            print(f"Now fitting task {task_name} with {fit_function_name} sample {sample}, attempt {count} ...")            
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
            if chi2 < chi2_reference + 60:
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
                plt.savefig(target_folder / f"{sample}__{count}_{DAY}.png",dpi=200, bbox_inches='tight', pad_inches=0.05)

            count += 1

    # =======================================================================================
    # 3. Ellipsoid sym (few attempts)
    # =======================================================================================
    task_name = "BXII_pure_ellips_sym"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()

    fit_function_name = "ellipsoid_sym_C4"
    fit_function = lambda q,  R,  sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale, c4: fit.ULV_ellipsoid(q, R,  sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale,   0, 0, c4)
    for sample in who:
        count = 0
        while (count < 3):
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

            if sample == "BXII_S1":
                R_max_lim = 20
                sigma_R_max_lim = 8
            elif sample == "BXII_S3":
                R_max_lim = 65
                sigma_R_max_lim = 35
            elif sample == "BXII_S4":
                R_max_lim = 115
                sigma_R_max_lim = 70
            
            # ========================== Setting Bounds ==============================
            # p0         = [         R,          sigma_R,     eps, sigma_eps,  rho_h, sigma_h,    sigma_c,      z_h,    scale,    c4] 
            lower_bounds = [        14,              1.0,     0.7,       0.0,    0.9,    0.10,        0.5,      1.7,        0,     0] 
            upper_bounds = [ R_max_lim,  sigma_R_max_lim,     1.2,       0.7,    2.5,     0.5,        0.9,      2.0,   0.0001,     1] 
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
            if chi2 < chi2_reference + 200:
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
                plt.savefig(target_folder / f"{sample}_{fit_function_name}_{count}_{DAY}.png",dpi=200, bbox_inches='tight', pad_inches=0.05)

            count += 1
    """

    # =======================================================================================
    # 4. Ellipsoid asym 
    # =======================================================================================
    task_name = "BXII_pure_ellips_ASYM"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()

    fit_function_name = "ellipsoid_asym_C4"
    fit_function = lambda q,  R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale, c4: fit.ULV_ellipsoid_asym(q, R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale,   0, 0, c4)
    for sample in who:
        count = 0
        while (count < 3):
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

            if sample == "BXII_S1":
                R_max_lim = 20
                sigma_R_max_lim = 8
            elif sample == "BXII_S3":
                R_max_lim = 65
                sigma_R_max_lim = 35
            elif sample == "BXII_S4":
                R_max_lim = 115
                sigma_R_max_lim = 70
            
            # ========================== Setting Bounds ==============================
            # ========================== Setting Bounds ==============================
            # p0         = [          R,          sigma_R,         eps,   sigma_eps,        rho_h1,      rho_h2,     sigma_h1,      sigma_h2,     sigma_c,          z_h,        scale,    c4]
            lower_bounds = [         13,                1,         0.7,           0,           0.8,         0.8,         0.13,          0.13,         0.5,          1.7,            0,     0] 
            upper_bounds = [  R_max_lim,  sigma_R_max_lim,         1.2,         0.7,           2.5,         2.5,         0.60,          0.60,         1.0,          2.0,         1e-1,     1] 
            p0 = np.random.uniform(lower_bounds, upper_bounds)

            print(f"Now fitting task {task_name} with {fit_function_name} sample {sample}, attempt {count} ...")            
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

            param_names = ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h1_fit", "rho_h2_fit",
                           "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit","scale_fit", "c4_fit"]

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
            if chi2 < chi2_reference + 200:
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
                plt.savefig(target_folder / f"{sample}_{fit_function_name}_{count}_{DAY}.png",dpi=200, bbox_inches='tight', pad_inches=0.05)

            count += 1
