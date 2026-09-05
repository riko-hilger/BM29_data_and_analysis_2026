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

def average_deviation_percent(a, b):
    deviations = [abs(x - y) / abs(x) * 100 for x, y in zip(a, b)]
    return sum(deviations) / len(deviations)
threshold = 10
# =============== SPHERIC ASYM NO BGR ======================================
fit_function_name = "ULV_spheric_asym_C4"
fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
DAY = "Sep1"
task_name = "sph_asym_ambiguity_"

who = [
    "BIX_S2",
    "BVI_S1",
    "BVI_S2",
    "BVI_S3",
    "BVI_S4",
    "BVI_S5",
    "BVI_S6",
    "BVI_S7",
    "BVI_S8",
    "BVI_S9",
    "BVI_S10",

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

this_file = Path(__file__).resolve()
analysis_BM29_July2026 = this_file.parent.parent
target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
if not target_folder.exists():
    target_folder.mkdir()

for sample in who:
    print(f"Now working on sample {BM29.sample_dict[sample]} ...")
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
    while count < 200:
        plt.close()
        # ========================== Setting Bounds ==============================
        # p0           = [     R,       sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      c4]
        lower_bounds = [       0,             0,      0.0,      0.0,       0.13,      0.13,      0.5,    1.5,       0,       0] #,       0,       0, -np.inf]
        upper_bounds = [  27,         10,      3.9,      3.9,         0.8,       0.8,      1.2,      2,      1,   1] #,       4,     0.5, +np.inf]
        p0 = np.random.uniform(lower_bounds, upper_bounds)
        
        # while count < 200:
        #     count += 1
        #     p0_EDP = np.random.uniform(lower_bounds[2:7], upper_bounds[2:7])
        #     p0 = np.concatenate(([R, sigma_R], p0_EDP, [z_h, scale, c4]))
            # print(p0)
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

        novelty_list = [average_deviation_percent(popt, previous_fit) for previous_fit in fit_list]
        if len(fit_list) == 0: 
            fit_list.append(popt)
            print(f"Found successful fit nr. {i} for {sample} at attempt {count} with chi2 = {chi2:.4g}")
            print(fit_lib_entry) 
            print("\n")          
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
        else:
            if all(novelty > threshold for novelty in novelty_list):
                fit_list.append(popt)
                print(f"Found successful fit nr. {i} for {sample} at attempt {count} with chi2 = {chi2:.4g}")
                print(fit_lib_entry)  
                print("\n")          
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
                                