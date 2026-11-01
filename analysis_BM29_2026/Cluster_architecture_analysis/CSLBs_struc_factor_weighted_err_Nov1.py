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

who = [
    # "BXIII_S2", #: "BXIII_S2_60%_CSLB",

    "BXIII_S12", #: "BXIII_S12_60%_CSLB_4uM_Syn1",
    "BXIII_S11", #: "BXIII_S11_60%_CSLB_6uM_Syn1",
    "BXIII_S13", #: "BXIII_S13_60%_CSLB_6uM_Syn1_12uM_aSynWT",
    "BXIII_S14", #: "BXIII_S14_60%_CSLB_6uM_Syn1_24uM_aSynWT",

    "BXIV_S3" #: "BXIV_S3_60%_CSLB_6uM_Syn1_24uM_ATP"
]

if __name__ == "__main__":
    # =============================================================================
    DAY = "Nov1"
    # =======================================================================================
    # 2. Fit total q-range with additional structure factor
    # =======================================================================================
    task_name = "CSLBs_struc_factor_weighted_err_LARGEsearch"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()

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
        mask = (q >= 0.0) & (q <= 1.2)
        q    = q[mask]
        I    = I[mask]
        I_err_Poisson = I_err_Poisson[mask]
        Ierr = BM29.CSLB_weighted_op_error(q, I)

        # ======================== keep track of fits =====================
        fit_list = []
        fits = BM29.best_fits[sample]["ellips_DAB_struc_factor"]
        for key in fits:
            fit_list.append(fits[key])

        ellipsoid_homogen_DAB_name = "ellipsoid_homogen_DAB"
        ellipsoid_homogen_DAB = lambda q, R, sigma_R, eps, sigma_eps, scale, c4, c, xi: fit.ellipsoid_homogen(q, R, sigma_R, eps, sigma_eps, scale, c4) + fit.DAB_model(q, c, xi)
        I_fit = ellipsoid_homogen_DAB(q, *BM29.best_fits[sample]["ellipsoid_homogen_DAB"][-1])

        fit_function_name = "ellips_DAB_struc_factor"
        fit_function = lambda q, A, xi, D, s, P, q0, w: I_fit * BM29.structure_factor_single_peak(q, A, xi, D, s, P, q0, w)
        param_names = ["A", "xi", "D", "s", "P", "q0", "w"]
        # A = 0.8     # A > 0 
        # xi = 20     # xi ≈ correlation length (~10–30 nm)
        # D = 0.2     # 
        # s = 0.15    # Width of the Gaussian centered at q = 0
        # P = 0.3     # Amplitude of the nearest-neighbour correlation peak
        # q0 = 0.17   # Position of the nearest-neighbour correlation peak
        # w = 0.1    # Width of the nearest-neighbour correlation peak

        # ========================== Setting Bounds ==============================
        # p0         = [           A,       xi,       D,       s,     P,       q0,         w]
        lower_bounds = [           0,      2.0,     -10,     0.0,   0.0,     0.06,       0.0]     
        upper_bounds = [          11,      100,     +10,     0.3,   3.0,     0.15,       0.7]

        count = 1
        while (count < 25001):
            if len(fit_list) > 15:
                break
            print(f"Now fitting task {task_name} with {fit_function_name} sample {sample}, attempt {count} ...")            
            if count==1:
                p0 = BM29.best_fits[sample]["ellips_DAB_struc_factor"][1]
                if all(lo <= v <= hi for v, lo, hi in zip(p0, lower_bounds, upper_bounds)):
                    p0 = p0
                else:
                    p0 = np.random.uniform(lower_bounds, upper_bounds)
            else:
                p0 = np.random.uniform(lower_bounds, upper_bounds)
            #======================== Fit ===========================================
            try:
                popt, pcov = curve_fit(
                    fit_function,
                    q,
                    I,
                    sigma=Ierr,
                    maxfev=10000,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds)
                )
            except RuntimeError:
                print("Maximum number of evaluations reached — skipping this fit.")
                continue
    
            perr = np.sqrt(np.diag(pcov))
            fit_lib_entry = "["
            for name, value in zip(param_names, popt):
                globals()[name] = value
                fit_lib_entry += f"{value:.4g}, "
            fit_lib_entry = fit_lib_entry[:-2]
            fit_lib_entry += "]"
            print(fit_lib_entry)
            chi2, info_text_print = fit.all_info(q, I, Ierr, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

            # chi2_reference = 20 # BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq")
            # if chi2 < chi2_reference + 200:
            chi2_threshold = 23
            if chi2 <= chi2_threshold:
                novelty_list = [average_deviation_percent(popt, previous_fit) for previous_fit in fit_list]
                if all(novelty > threshold for novelty in novelty_list):
                    fit_list.append(popt)
                    print("Fit successful with chi2 = ", chi2)
                    # print(info_text_print)
                    with open(target_folder/f"{sample}_{DAY}.txt", "a", encoding="utf-8") as k:
                        k.write(f"\n\n{BM29.sample_dict[sample]} fitted (count={count}) using {fit_function_name} with chi2 = {chi2}\n" + fit_lib_entry + "\n###########################################\n")
                    #=========================== Plot ===========================================
                    plt.close("all")
                    plt.figure(figsize=(12, 7.5))
                    plt.errorbar(q, I, Ierr, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
                    plt.loglog(q, fit_function(q, *popt), color='red', label='Fit')
                    plt.loglog(q, fit_function(q, *p0), color='orange', label='initial guess')

                    plt.xlabel('$q [\\mathrm{nm}^{-1}]$')
                    plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
                    plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name}')
                    plt.grid(True)
                    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
                    legend.set_title(info_text_print)
                    plt.tight_layout()
                    plt.savefig(target_folder / f"{sample}__{count}_{DAY}.png",dpi=100, bbox_inches='tight', pad_inches=0.05)
                    g = BM29.plot_structure_factor_single_peak([sample], initial_guess=p0)
                    g.savefig(target_folder / f"{sample}__{count}_{DAY}_structure_factor.png", dpi=100, bbox_inches='tight', pad_inches=0.05)
            count += 1

