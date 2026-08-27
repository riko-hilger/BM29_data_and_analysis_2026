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
# =========================================================================
fit_function_name = "ULV_spheric_asym_C4"
fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
DAY = "Aug23"
task_name = "spheric_asym_EXPLORE_AMBIGUITY_op_error"

who = [
    #"BII_S9",

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

for sample in who:
    print(f"Now working on sample {BM29.sample_dict[sample]} ...")
    # -------------- LOAD DATA --------------------------------------------------
    block = sample.split("_")[0]
    block_path = BM29.blockpath(block)
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
 
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()

    dir_names = [x.name for x in block_path.iterdir()]
    for dir_name in dir_names:
        if dir_name[:-13].endswith(sample):
            sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
            q, I, Ierr = np.loadtxt(sub_dat_path, unpack=True)

    q     = q
    I     = I
    I_err_op = BM29.op_error_classic(q, I)

    # ========================== Setting Bounds ==============================
    lower_bounds = [   9,          2,      1.0,      1.0,       0.13,      0.13,      0.5,    1.5,       0,       0]
    upper_bounds = [  25,          9,      2.5,      2.5,        0.7,       0.7,      1.0,      2,   0.001,       1] 

    fit_list = []
    fit_list.append(BM29.best_fits[sample]["ULV_spheric_asym_C4"][1])

    count = 0
    chi2 = 1e6
    i = 2
    while ((i < 11) and (count < 500)):
        count += 1
        if count % 500 == 0:
            print(f"Reached attempt {count}...")
        p0 = np.random.uniform(lower_bounds, upper_bounds)
        #======================== Fit ===========================================
        popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_op, maxfev=10000, p0=p0, bounds = (lower_bounds, upper_bounds))

        param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit", "z_h_fit", "scale_fit", "c4_fit"]
        perr = np.sqrt(np.diag(pcov))
        fit_lib_entry = "["
        for name, value in zip(param_names, popt):
            globals()[name] = value
            fit_lib_entry += f"{value:.4g}, "
        fit_lib_entry = fit_lib_entry[:-2]
        fit_lib_entry += "]"
        chi2, info_text_print = fit.all_info(q, I, I_err_op, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

        if chi2 < 26: 
            novelty_list = [average_deviation_percent(popt, previous_fit) for previous_fit in fit_list]
            if all(novelty > threshold for novelty in novelty_list):
                fit_list.append(popt)
                print(f"Found successful fit nr. {i} for {sample} at attempt {count} with chi2 = {chi2:.4g}")
                print(fit_lib_entry)  
                print("\n")          
                with open(target_folder/f"info_texts_{sample}.txt", "a", encoding="utf-8") as k:
                    k.write(f"\n\n{BM29.sample_dict[sample]} using {fit_function_name}, nr. {i}\n" + fit_lib_entry + "\n###########################################\n")
                i += 1

                
