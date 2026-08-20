from multiprocessing.pool import Pool
from pathlib import Path
import sys
from functools import partial
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
import fit_lib as fit
import BM29_lib as BM29
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = False
from scipy import optimize
from scipy.optimize import curve_fit

DAY = "Aug19"

fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c4: fit.ULV_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, 0, 0, c4)
fit_function_name = "ULV_sph_asym_prot_layer_C4"

this_file = Path(__file__).resolve()
analysis_BM29_July2026 = this_file.parent.parent
target_folder = analysis_BM29_July2026 / "Results_and_Figures" / DAY / f"fit_{fit_function_name}"
if not target_folder.exists():
    target_folder.mkdir()

who = [
    # "BII_S9",

    # LV4 + aSyn WT
    # "BIV_S1",
    # "BIV_S2",
    # "BIV_S3",
    # "BIV_S4",
    # "BIV_S5",
    # "BIV_S6",
    "BIV_S9",
    "BIV_S10",
    "BIV_S7",
    "BIV_S8",

    # LV4 + aSyn A124T
    # "BIV_S21",
    # "BIV_S22",
    # "BIV_S23",
    # "BIV_S24",
    # "BIV_S25",
    # "BIV_S26",
    "BIV_S27",
    "BIV_S28",
    "BIV_S29",
    "BIV_S30",  
]

if __name__ == "__main__":
    for sample in who:
        block = sample.split("_")[0]
        block_path = BM29.blockpath(block)
        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / BM29.sample_dict[sample]
                q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)
        plt.close("all")

        q     = q
        I     = I
        I_err_Poisson_impro = BM29.improved_Poisson_error_low_high(q, Ierr_Poisson)

        # ========================== Parameters ==========================
        # if sample == "BII_S9":
        #     p0 = [9.9997, 5.9694, 1.1317, 0.10304, 1.8799, 1.0389, 0.30377, 0.44115, 0.89864, 1.7498, 6.2771e-06]
        # elif "1uM" in BM29.sample_dict[sample]:        
        #     p0 = [9.3033, 6.0629, 1.1275, 0.11246, 1.2544, 1.5019,  0.35065,  0.20000,  0.608, 1.8441, 0.00021972]
        # elif "2uM" in BM29.sample_dict[sample]:        
        #     p0 = [9.3033, 6.0629, 1.1275, 0.11246, 1.2544, 1.5019,  0.35065,  0.20000,  0.608, 1.8441, 0.00021972]
        # elif "3uM" in BM29.sample_dict[sample]:  
        #     p0 = [10.497, 6.4909, 1.083, 0.24643, 1.4598, 1.7853,  0.34637,  0.20003,  0.7235,  1.8187, 0.00018308]      
        # elif "6uM" in BM29.sample_dict[sample]:     
        #     p0 =   [10.615, 6.2472, 1.1741, 0.21722, 1.478,  1.7475,  0.36964,  0.20011,  0.73867, 1.7677] 
        # elif "9uM" in BM29.sample_dict[sample]:       
        #     p0 =  [10.871, 7.0448, 0.99143, 0.21195, 1.3512, 1.8002,  0.37945,  0.20025,  0.74247, 1.8666, 8.026e-6]
        # elif "15uM" in BM29.sample_dict[sample]:
        #     p0 = [10.847, 6.0374, 1.0642, 0.24972, 1.5328, 1.6601,  0.35989,  0.20032,  0.71574, 1.8356, 7.303e-6]
        # elif "25uM" in BM29.sample_dict[sample]:
        #     p0 = [10.471, 7.133, 0.98558, 0.21418, 1.4986, 1.7302,  0.37497,  0.20002,  0.75887, 1.8633, 8.08e-6]
        # if "50uM" in BM29.sample_dict[sample]:
        #     p0 = params = [9.7611, 8.0389, 2.8854, 0.77629, 0.2, 0.37505, 1.0, 1.5, 2.4493, 0.2, 1.2342, 6.6553e-07]
        # elif "75uM" in BM29.sample_dict[sample]:
        #     p0 = [9.7611, 8.0389, 2.8854, 0.77629, 0.2, 0.37505, 1.0, 1.5, 2.4493, 0.2, 1.2342, 6.6553e-07]
        # elif "100uM" in BM29.sample_dict[sample]:
        #     p0 = [9.7611, 8.0389, 2.8854, 0.77629, 0.2, 0.37505, 1.0, 1.5, 2.4493, 0.2, 1.2342, 6.6553e-07]

        # ========================== Setting Bounds ==============================
        # p0         = [        R,  sigma_R,       rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,  z_h3, sigma_h3,   rho_h3,     scale,       c4]
        lower_bounds = [        9,        4,          0.8,    0.8,       0.12,     0.12,      0.5,      1.7,     3,      0.2,      0.0,         0,        0] #,    0,    0,  -np.inf]
        upper_bounds = [       15,        9,          3.7,    3.7,        0.6,      0.6,      1.2,        2,     5,        2,      2.0,      1e-5,  +np.inf] # ,  0.2,  0.5,  +np.inf]
        p0           =     [14.31,    8.624,       0.9992,    3.5,     0.5716,   0.1706,    1.107,    1.794, 3.634,      1.9,  0.04283, 2.628e-07,        0] #,    0,    0,  -np.inf]

        #======================== Fit ===========================================
        popt, pcov = curve_fit(fit_function, q, I, sigma=I_err_Poisson_impro, maxfev=10000, p0=p0, bounds = (lower_bounds, upper_bounds))

        param_names = ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit",
                    "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit", "z_h3_fit", "sigma_h3_fit", "rho_h3_fit", "scale_fit", "c4_fit"]

        perr = np.sqrt(np.diag(pcov))
        fit_lib_entry = "["
        for name, value in zip(param_names, popt):
            globals()[name] = value
            fit_lib_entry += f"{value:.4g}, "
        fit_lib_entry = fit_lib_entry[:-2]
        fit_lib_entry += "]"
        print(f"Successful fit for {BM29.sample_dict[sample]} with fitted params:")
        print(fit_lib_entry)
        chi2, info_text_print = fit.all_info(q, I, I_err_Poisson_impro, param_names, p0, popt, perr, fit_function, lower_bounds, upper_bounds)

        print(info_text_print)
        with open(target_folder/"info_texts_Aug19.txt", "a", encoding="utf-8") as k:
            k.write(f"\n\n{BM29.sample_dict[sample]} using spherical asym + protein layer model\n" + info_text_print + "\n" + fit_lib_entry + "\n###########################################\n")
        # #=========================== Plot ===========================================

        q_fit = np.linspace(min(q), max(q), len(q))
        plt.figure(figsize=(12, 7.5))
        plt.errorbar(q, I, I_err_Poisson_impro, marker="o", markersize=2,markeredgewidth=1,markeredgecolor='Blue', markerfacecolor='None', alpha=0.85, ls="None", label='I_SUB', zorder=1)
        plt.loglog(q_fit, fit_function(q_fit, *popt), color='red', label='Fit')
        plt.loglog(q_fit, fit_function(q_fit, *p0), color='orange', label='initial guess')
        plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, c4_fit), color='black', label='pow law bgr')
        plt.loglog(q_fit, fit.powerlaw_bgr(q_fit, 0, 0, p0[-1]), color='black', label='initial bgr', ls="--")

        plt.xlabel('$q [1/\\mathrm{nm}]$')
        plt.ylabel('$I(q) [\\mathrm{cm}^{-1}]$')
        plt.title(f'{BM29.sample_dict[sample]} \nusing sph. asym. prot. layer (improv. Poisson err.)')
        plt.grid(True)
        plt.legend()
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
        legend.set_title(info_text_print)

        plt.tight_layout()
        plt.savefig(target_folder / f"{BM29.sample_dict[sample]}_Aug19.png", bbox_inches='tight', pad_inches=0.05)
        #==================== EDP Plot ==================================================
        fig=fit.EDP_plot(R_fit, sigma_h1_fit, sigma_h2_fit, sigma_c_fit, rho_h1_fit, rho_h2_fit, z_h_fit, z_h_fit, info_text_print,  f'{BM29.sample_dict[sample]} using {fit_function_name}', z_h3_fit, sigma_h3_fit, rho_h3_fit)
        fig.savefig(target_folder / f"EDP_{BM29.sample_dict[sample]}_Aug19.png", dpi=400, bbox_inches='tight', pad_inches=0.05)

        
        



