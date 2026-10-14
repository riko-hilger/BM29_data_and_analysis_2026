from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import fit_lib as fit
import BM29_lib as BM29
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = False
from scipy.optimize import curve_fit

def average_deviation_percent(a, b):
    deviations = [abs(x - y) / abs(x) * 100 for x, y in zip(a, b)]
    return sum(deviations) / len(deviations)

threshold = 10

# =============== SPHERIC ASYM NO BGR ======================================
fit_function_name = "ULV_spheric_asym_C4"
fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c4: fit.ULV_spheric_asym(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, 0, 0, c4)

DAY = "Sep13"
task_name = "sph_asym_ambiguity_BVII"

who = [
    "BII_S10",
    "BII_S11",

    "BVII_S1",
    "BVII_S2",
    "BVII_S3",
    "BVII_S4",
    "BVII_S5",
    "BVII_S6",
    "BVII_S7",
    "BVII_S8",
    "BVII_S9",
    "BVII_S10",
    "BVII_S11",
    "BVII_S12",
    "BVII_S13",
    "BVII_S14",
    "BVII_S15",
    "BVII_S16",
    "BVII_S17",
    "BVII_S18",
    "BVII_S19",
    "BVII_S20",
    "BVII_S21",
    "BVII_S22",

    "BIX_S10",
    "BIX_S11",
    "BIX_S12",
    "BIX_S13",
    "BIX_S14",
    "BIX_S15",
    "BIX_S16",
    "BIX_S17",
    "BIX_S18",
    "BIX_S19",
]

this_file = Path(__file__).resolve()
analysis_BM29_July2026 = this_file.parent.parent
target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
if not target_folder.exists():
    target_folder.mkdir(parents=True)

param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit", "z_h_fit", "scale_fit", "c4_fit"]

for sample in who:
    print(f"\nNow working on sample {BM29.sample_dict[sample]} ...")

    # -------------- LOAD DATA --------------------------------------------------
    block = sample.split("_")[0]
    block_path = BM29.blockpath(block)
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name / sample
    if not target_folder.exists():
        target_folder.mkdir(parents=True)

    dir_names = [x.name for x in block_path.iterdir()]
    for dir_name in dir_names:
        if dir_name[:-13].endswith(sample):
            sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
            q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)

    mask = (q >= 0.07) & (q <= 5.1)
    q = q[mask]
    I = I[mask]
    Ierr_Poisson = Ierr_Poisson[mask]
    I_err_op = BM29.Poisson_error_focus_minimum_highq(q, Ierr_Poisson)

    # ======================================================================
    # 1. FIND BEST FIT WITH 200 RANDOM INITIAL GUESSES
    # ======================================================================
    best_popt = None
    best_perr = None
    best_p0 = None
    best_chi2 = np.inf
    best_info_text_print = None
    best_fit_lib_entry = None

    count = 0
    while count < 200:
        count += 1
        plt.close()

        # p0 = [R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c4]
        lower_bounds = [8,   0, 0.7, 0.7, 0.13, 0.13, 0.5, 1.5, 0, 0]
        upper_bounds = [27, 10, 3.5, 3.5,  0.6,  0.6, 1.2,   2, 1, 1]
        p0 = np.random.uniform(lower_bounds, upper_bounds)

        try:
            popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_op, maxfev=10000, p0=p0, bounds=(lower_bounds, upper_bounds))
        except (RuntimeError, ValueError) as error:
            print(f"Fit attempt {count} failed for {sample}: {error}")
            continue

        perr = np.sqrt(np.diag(pcov))
        fit_lib_entry = "["
        for name, value in zip(param_names, popt):
            globals()[name] = value
            fit_lib_entry += f"{value:.4g}, "
        fit_lib_entry = fit_lib_entry[:-2] + "]"

        chi2, info_text_print = fit.all_info(q, I, I_err_op, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)
        current_chi2 = BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq", popt)

        if current_chi2 < best_chi2:
            best_chi2 = current_chi2
            best_popt = popt.copy()
            best_perr = perr.copy()
            best_p0 = p0.copy()
            best_info_text_print = info_text_print
            best_fit_lib_entry = fit_lib_entry

    # ======================================================================
    # PRINT AND SAVE BEST FIT
    # ======================================================================
    print(f"\nBEST FIT for {sample}:")
    print(f"chi2 = {best_chi2:.4g}")
    print(best_fit_lib_entry)

    with open(target_folder / f"info_texts_{sample}.txt", "a", encoding="utf-8") as k:
        k.write(f"\n\n{BM29.sample_dict[sample]} BEST FIT with chi2 = {best_chi2:.4g}\n" + best_fit_lib_entry + "\n###########################################\n")

    plt.figure(figsize=(12, 7.5))
    plt.errorbar(q, I, I_err_op, marker="o", markersize=2, markeredgewidth=1, ecolor="Blue", markeredgecolor="Blue", markerfacecolor="None", alpha=0.5, ls="None", label="I_SUB", zorder=1)
    plt.loglog(q, fit_function(q, *best_popt), color="red", label="Best fit")
    plt.loglog(q, fit_function(q, *best_p0), color="orange", label="initial guess")
    plt.xlabel('$q [1/\\mathrm{nm}]$')
    plt.ylabel('$I(q)$')
    plt.title(f'{sample} using spheric_asym + C4, best fit')
    plt.grid(True)
    legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), alignment="left")
    legend.set_title(best_info_text_print)
    plt.tight_layout()
    plt.savefig(target_folder / f"{fit_function_name}_best.png", dpi=200, bbox_inches="tight", pad_inches=0.05)

    # ======================================================================
    # 2. EXPLORE FIT AMBIGUITY
    #    BEST FIT IS THE FIRST ENTRY IN fit_list
    # ======================================================================
    fit_list = [best_popt.copy()]
    best_chi2_reference = best_chi2
    chi2_threshold = best_chi2_reference + 20
    i = 1
    count = 0

    print(f"\nStarting ambiguity search for {sample}")
    print(f"Reference chi2 = {best_chi2_reference:.4g}")
    print(f"Accepted chi2 threshold = {chi2_threshold:.4g}")

    # ========================== Setting Bounds ==============================
    for sigma_h2 in [0.2, 0.4]:
        for R_lowlim in [9, 11, 16]:
            for r_uplim in [1.4, 1.6, 2.5]:
                for s_uplim in [0.55, 0.8]:
                    for s_lowlim in [0, 0.1, 0.15, 0.20]:
                        for r_lowlim in [0.7, 0.8, 1, 1.2]:
                            count += 1
                            plt.close()

                            lower_bounds = [R_lowlim, 1, r_lowlim, r_lowlim, s_lowlim, s_lowlim, 0.5, 1.6, 0, 0]
                            upper_bounds = [27, 8.5, r_uplim, r_uplim, s_uplim, s_uplim, 1, 2, 1e-3, 1e-2]
                            p0 = np.random.uniform(lower_bounds, upper_bounds)

                            try:
                                popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_op, maxfev=10000, p0=p0, bounds=(lower_bounds, upper_bounds))
                            except (RuntimeError, ValueError) as error:
                                print(f"Ambiguity fit attempt {count} failed for {sample}: {error}")
                                continue

                            perr = np.sqrt(np.diag(pcov))
                            fit_lib_entry = "["
                            for name, value in zip(param_names, popt):
                                globals()[name] = value
                                fit_lib_entry += f"{value:.4g}, "
                            fit_lib_entry = fit_lib_entry[:-2] + "]"

                            chi2, info_text_print = fit.all_info(q, I, I_err_op, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)
                            current_chi2 = BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq", popt)
                            novelty_list = [average_deviation_percent(popt, previous_fit) for previous_fit in fit_list]

                            if all(novelty > threshold for novelty in novelty_list) and current_chi2 < chi2_threshold:
                                fit_list.append(popt.copy())

                                print(f"Found ambiguous fit nr. {i} for {sample} at attempt {count} with chi2 = {current_chi2:.4g}")
                                print(fit_lib_entry)

                                with open(target_folder / f"info_texts_{sample}.txt", "a", encoding="utf-8") as k:
                                    k.write(f"\n\n{BM29.sample_dict[sample]} ambiguity fit nr. {i} with chi2 = {current_chi2:.4g}\n" + fit_lib_entry + "\n###########################################\n")

                                plt.figure(figsize=(12, 7.5))
                                plt.errorbar(q, I, I_err_op, marker="o", markersize=2, markeredgewidth=1, ecolor="Blue", markeredgecolor="Blue", markerfacecolor="None", alpha=0.5, ls="None", label="I_SUB", zorder=1)
                                plt.loglog(q, fit_function(q, *popt), color="red", label="Fit")
                                plt.loglog(q, fit_function(q, *p0), color="orange", label="initial guess")
                                plt.xlabel('$q [1/\\mathrm{nm}]$')
                                plt.ylabel('$I(q)$')
                                plt.title(f'{sample} using spheric_asym + C4, fit nr. {i}')
                                plt.grid(True)
                                legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), alignment="left")
                                legend.set_title(info_text_print)
                                plt.tight_layout()
                                plt.savefig(target_folder / f"{fit_function_name}_{i}.png", dpi=200, bbox_inches="tight", pad_inches=0.05)

                                i += 1

    print(f"\nFinished ambiguity search for {sample}: {len(fit_list)} distinct fits found including the best fit.")
    print(f"Best fit chi2 = {best_chi2:.4g}")
    print(f"Total ambiguity attempts = {count}")