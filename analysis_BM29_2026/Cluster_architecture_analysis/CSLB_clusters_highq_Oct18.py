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
    "BXIII_S2", #: "BXIII_S2_60%_CSLB",

    "BXIII_S12", #: "BXIII_S12_60%_CSLB_4uM_Syn1",
    "BXIII_S11", #: "BXIII_S11_60%_CSLB_6uM_Syn1",
    "BXIII_S13", #: "BXIII_S13_60%_CSLB_6uM_Syn1_12uM_aSynWT",
    "BXIII_S14", #: "BXIII_S14_60%_CSLB_6uM_Syn1_24uM_aSynWT",

    "BXIV_S3" #: "BXIV_S3_60%_CSLB_6uM_Syn1_24uM_ATP"
]

"""
    1. Fit high-q with ellipsoid_homogen + DAB
    2. Fit total q-range with additional structure factor
"""

if __name__ == "__main__":
    # =============================================================================
    DAY = "Oct18"
    # =======================================================================================
    # 1. Fit high-q with ellipsoid_homogen + DAB
    # =======================================================================================
    task_name = "CSLB_clusters_highq"
    this_file = Path(__file__).resolve()
    analysis_BM29_July2026 = this_file.parent.parent
    target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name
    if not target_folder.exists():
        target_folder.mkdir()
    fit_function_name = "ellipsoid_homogen_DAB"
    fit_function = lambda q, R, sigma_R, eps, sigma_eps, scale, c4, c, xi: fit.ellipsoid_homogen(q, R, sigma_R, eps, sigma_eps, scale, c4) + fit.DAB_model(q, c, xi)
    param_names = ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "scale_fit", "c4_fit", "c_fit", "xi_fit"]

    for sample in who:
        this_file = Path(__file__).resolve()
        analysis_BM29_July2026 = this_file.parent.parent
        target_path = analysis_BM29_July2026 / "Results_and_Figures" / DAY / task_name / sample
        if not target_path.exists():
            target_path.mkdir()

        # ========================== Setting Bounds ==============================
        # p0         = [           R,         sigma_R,          eps, sigma_eps,     scale,       c4,        c,       xi]
        # p_CSLB_pure= [       27.19,          0.4084,       0.9592,    0.1784, 7.098e-09, 0.001004, 0.003202,    1.203]
        lower_bounds = [          26,             0.2,          0.8,       0.0,       0.0,      0.0,      0.0,      1.0] 
        upper_bounds = [          30,             5.0,          1.2,       0.5,      1e-6,     0.01,      2.0,      1.4] 

        qmin_dict = {0.25: "0p25", 0.32: "0p32", 0.4: "0p4"}
        for q_min in qmin_dict.keys():
            target_folder = target_path / qmin_dict[q_min]
            if not target_folder.exists():
                    target_folder.mkdir()

            block = sample.split("_")[0]
            block_path = BM29.blockpath(block)
            dir_names = [x.name for x in block_path.iterdir()]
            for dir_name in dir_names:
                if dir_name[:-13].endswith(sample):
                    sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
                    q, I, I_err_Poisson = np.loadtxt(sub_dat_path, unpack=True)
            q_full = np.copy(q[q <= 1.2])
            I_full = np.copy(I[q <= 1.2])
            I_err_full = BM29.CSLB_op_error(q_full, I_full)
            mask = (q >= q_min) & (q <= 1.2)
            q    = q[mask]
            I    = I[mask]
            I_err_Poisson = I_err_Poisson[mask]

            Ierr = BM29.CSLB_op_error(q, I)

            thinning = 1
            q_thinned = q#[::thinning]
            I_thinned = I#[::thinning]
            I_err_thinned = Ierr#[::thinning]
            count = 0
            while (count < 3):
                p0 = np.random.uniform(lower_bounds, upper_bounds)
                print(f"Now fitting task {task_name} with {fit_function_name} sample {sample}, qmin {q_min}, {count} ...")            
                #======================== Fit ===========================================
                try:
                    popt, pcov = curve_fit(
                        fit_function,
                        q_thinned,
                        I_thinned,
                        sigma=I_err_thinned,
                        maxfev=7000,
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
    
                print("Fit successful with chi2 = ", chi2)
                # print(info_text_print)
                with open(target_folder/f"{sample}_{DAY}.txt", "a", encoding="utf-8") as k:
                    k.write(f"\n\n{BM29.sample_dict[sample]} fitted (count={count}) using {fit_function_name} with chi2 = {chi2}\n" + fit_lib_entry + "\n###########################################\n")
    
                chi2_reference = 20 # BM29.get_chi2(sample, "Poisson_error_focus_minimum_highq")
                if chi2 < chi2_reference + 0:
                    #=========================== Plot ===========================================
                    q_fit = np.linspace(min(q_full), max(q_full), 1000)
                    plt.close("all")
                    plt.figure(figsize=(12, 7.5))
                    plt.errorbar(q_full, I_full, I_err_full, marker="o", markersize=3,markeredgewidth=1,ecolor="Blue",      markeredgecolor='Blue',      markerfacecolor='None', alpha=0.5, ls="None", label='I_SUB', zorder=1)
                    plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
                    plt.loglog(q_fit, fit_function(q_fit, *p0), color='orange', label='initial guess')
                    plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, c4_fit), color='black', label='pow law bgr')
                    plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, p0[-1]), color='black', label='initial bgr', ls="--")
                    plt.axvline(q_min,color="grey",linestyle="--",linewidth=2.6,label=fr"$q_\mathrm{{min}}={q_min}$ nm$^{{-1}}$")

                    plt.xlabel('$q [\\mathrm{nm}^{-1}]$')
                    plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
                    plt.title(f'{BM29.sample_dict[sample]} \nusing {fit_function_name}')
                    plt.grid(True)
                    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
                    legend.set_title(info_text_print)
                    plt.tight_layout()
                    plt.savefig(target_folder / f"{sample}__{count}_{DAY}.png",dpi=100, bbox_inches='tight', pad_inches=0.05)
    
                count += 1

