import fit_lib as fit
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = False
import re
from scipy.stats import lognorm
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from scipy.stats import f
from scipy import optimize
from scipy.optimize import curve_fit

# COLORS
color_dict = {
    "pure_lipid": "grey",
    "WT": "dodgerblue",
    "A124T": "crimson",
    "E46K": "darkorange",
    "E61V": "green",
}

def extract_sample_name(sample):
    sample = sample[:-13] + "_"
    match = re.search(r"sample_([^_]+_[^_]+)", sample)
    result = match.group(1)
    block = result.split("_")[0]
    return result, block

def blockpath(block):
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent      
    data_root = project_root / "data_BM29_2026"
    
    return data_root / block

# ====================================================================================================
def gaussian(r, rho_pos, sigma, rho_h):
    return rho_h * np.exp(-0.5 * ((r - rho_pos) / sigma)**2)

def EDP_plot_together(who, title, fit_function="ULV_spheric_asym_C4"):
    norm = LogNorm(vmin=1, vmax=100)
    c_min = 0
    c_max = 100
    f=plt.figure(figsize=(12, 7.5))
    for sample in who:
        if "WT" in sample_dict[sample]:
            cmap = plt.cm.Blues
            variant = "WT"
        elif "A124T" in sample_dict[sample]:
            cmap = plt.cm.Reds
            variant = "A124T"
        elif "E46K" in sample_dict[sample]:
            cmap = plt.cm.Oranges
            variant = "E46K"
        elif "E61V" in sample_dict[sample]:
            cmap = plt.cm.Greens
            variant = "E61V"

        if concentrations[sample] == 0:
            color = "grey"
            variant = "pure_lipid"

        else:
            # Same color normalization as BIV_plot_together()
            norm = (
                concentrations[sample] - c_min
            ) / (
                c_max - c_min
            )

            # Avoid extremely pale colors
            norm = 0.25 + 0.70 * norm

            color = cmap(norm)
        R = 0
        if ((fit_function == "ULV_spheric_asym_C4") or (fit_function == "ULV_spheric_asym_C4_lognormal")):
            _, _,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   _, _ = best_fits[sample][fit_function][1]
            r = np.linspace(R - 2*z_h, R + 3*z_h, 1500)
            rho_h3, z_h3, sigma_h3 = 0, 1, 1
            rho_in, z_in, sigma_in = 0, 1, 1
        elif ((fit_function == "ULV_sph_asym_prot_layer") or (fit_function == "ULV_lognormal_sph_asym_prot_layer")):
            _, _, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, _, _ = best_fits[sample][fit_function][1]
            rho_in, z_in, sigma_in = 0, 1, 1
            r = np.linspace(R - 2*z_h, R + 4*z_h, 1500)
        elif fit_function == "ULV_lognormal_sph_asym_TWO_prot_layers":
            _, _, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_out, sigma_out, rho_out, z_in, sigma_in, rho_in, _, _ = best_fits[sample][fit_function][1]
            rho_h3, z_h3, sigma_h3 = rho_out, z_out, sigma_out
            r = np.linspace(R - 3.25*z_h, R + 3.25*z_h, 1500)
        rho_h   = [rho_in, rho_h1, -1, rho_h2, rho_h3]
        rho_pos = [R - z_in, R - z_h, R, R + z_h, R + z_h3]
        sigma   = [sigma_in, sigma_h1, sigma_c, sigma_h2, sigma_h3]
        rho = np.zeros_like(r)
        for i in range(5):
            rho += gaussian(r, rho_pos[i], sigma[i], rho_h[i])
        # color = color_dict[variant]
        if "WT" in sample_dict[sample]:
            plt.plot(r, rho, label=nice_sample_names[sample], color=color, linewidth=3, zorder=10)
        else:
            plt.plot(r, rho, label=nice_sample_names[sample], color=color, linewidth=3)

    plt.xlabel('$r[\\mathrm{nm}]$', fontsize=25)
    plt.ylabel('$\\rho\\mathrm{[a.u.]}$', fontsize=25)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.title(title)
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
    plt.close("all")
    return f

def plot_together(who, title, min_cutoff= 0.07, max_cutoff=5.1):
    """
    who: list of samples of the form "BII_S3"
    """
    plt.close("all")
    
    f = plt.figure(figsize=(14, 9))
    for sample in who:
        block = sample.split("_")[0]
        block_path = blockpath(block)
        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / sample_dict[sample]
                q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)
                mask = (q >= min_cutoff) & (q <= max_cutoff)
                q    = q[mask]
                I    = I[mask]
                Ierr_Poisson = Ierr_Poisson[mask]
                plt.errorbar(q, I, Ierr_Poisson, ls = "None", marker="o", label=nice_sample_names[sample], markersize=4.0)
    plt.xlabel("$q$")
    plt.ylabel("$I$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.title(title)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
    return f

# def plot_together_lognorm_polydisp(who):

#     plt.close("all")

#     f = plt.figure(figsize=(12, 7.5))
#     norm = LogNorm(vmin=1, vmax=100)
#     # norm = Normalize(vmin=1, vmax=100)
#     for sample in who:
#         if "WT" in sample_dict[sample]:
#             cmap = plt.cm.Blues
#         elif "A124T" in sample_dict[sample]:
#             cmap = plt.cm.Reds
#         elif "E46K" in sample_dict[sample]:
#             cmap = plt.cm.Oranges
#         elif "E61V" in sample_dict[sample]:
#             cmap = plt.cm.Greens
#         R0, sigma = best_fits[sample]["ULV_spheric_asym_C4_lognormal"][1][0:2]
#         R = np.linspace(0, R0 + 40, 1000)
#         P = lognorm.pdf(R, s=sigma, scale=R0)
#         # 0 µM aSyn = grey
#         if concentrations[sample] == 0:
#             color = "grey"
#         else:
#             color = cmap(norm(concentrations[sample]))
#         plt.plot(R, P, color=color, label=nice_sample_names[sample], lw=2.5)

#     plt.xlabel("$R$ [nm]")
#     plt.ylabel("$P(R)$")
#     plt.grid()
#     # plt.title("Lognormal polydispersity from fits with spherical asym. model")
#     legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), alignment="left")
#     return f

def normalized_gaussian(x, mu=0.0, sigma=1.0):
    """Normalized Gaussian (probability density): integrates to 1."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def plot_together_lognorm_polydisp(who, fit_function_name="ULV_spheric_asym_C4_lognormal"):

    plt.close("all")

    f = plt.figure(figsize=(12, 7.5))

    c_min = 0
    c_max = 100

    for sample in who:

        if "WT" in sample_dict[sample]:
            cmap = plt.colormaps["Blues"]
        elif "A124T" in sample_dict[sample]:
            cmap = plt.colormaps["Reds"]
        elif "E46K" in sample_dict[sample]:
            cmap = plt.colormaps["Oranges"]
        elif "E61V" in sample_dict[sample]:
            cmap = plt.colormaps["Greens"]

        R0, sigma = best_fits[sample][fit_function_name][1][0:2]

        R = np.linspace(0, R0 + 40, 1000)
        if ((fit_function_name == "ULV_spheric_asym_C4_lognormal") or (fit_function_name == "ULV_lognormal_sph_asym_TWO_prot_layers")):
            P = lognorm.pdf(R, s=sigma, scale=R0)
        elif fit_function_name == "ULV_spheric_asym_C4":
            P = normalized_gaussian(R, mu=R0, sigma=sigma)

        # 0 µM aSyn = grey
        if concentrations[sample] == 0:
            color = "grey"

        else:
            # Same color normalization as BIV_plot_together()
            norm = (
                concentrations[sample] - c_min
            ) / (
                c_max - c_min
            )

            # Avoid extremely pale colors
            norm = 0.25 + 0.70 * norm

            color = cmap(norm)

        plt.plot(
            R,
            P,
            color=color,
            label=sample_dict[sample],
            lw=4
        )

    plt.xlabel("$R$ [nm]", fontsize=30)
    plt.ylabel("$P(R)$", fontsize=30)
    # plt.title(title)
    plt.grid(True)
    tick_params = {'labelsize': 30}
    plt.tick_params(axis='both', **tick_params)

    legend = plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        alignment="left"
    )

    return f

# ===================================================================================================
def plot_structure_factor_single_peak(who, initial_guess=None, qmax=1):
    f=plt.figure(figsize=(12, 7.5))
    for sample in who:
        block = sample.split("_")[0]
        block_path = blockpath(block)
        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / sample_dict[sample]
                q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)
        mask = (q >= 0.07) & (q <= qmax)
        q    = q[mask]
        if initial_guess is not None:
            struc_fac_init = structure_factor_single_peak(q, *initial_guess)
            plt.plot(q, struc_fac_init, label="initial guess " + sample_dict[sample])
        structure_factor = structure_factor_single_peak(q, *best_fits[sample]["ellips_DAB_struc_factor"][1])
        plt.plot(q, structure_factor, label=sample_dict[sample])
    plt.xlabel("$q$ [nm$\mathrm{^{-1}}]$", fontsize=25)
    plt.ylabel("$S(q)$", fontsize=25)
    plt.grid()
    plt.legend()
    return f

def structure_factor_single_peak(q, A, xi, D, s, P, q0, w):
    return (1 + A/(1+(xi*q)**2)                 # Lorentzian (Ornstein-Zernike)
              - D*np.exp(-q**2/(2*s**2))        # Gaussian centered at q = 0  
              + P*np.exp(-(q-q0)**2/(2*w**2))  # Gaussian at q₀, nearest-neighbour correlation peak
    )
# A = 0.8     # A > 0 
# xi = 20     # xi ≈ correlation length (~10–30 nm)
# D = 0.2     # 
# s = 0.15    # Width of the Gaussian centered at q = 0
# P = 0.3     # Amplitude of the nearest-neighbour correlation peak
# q0 = 0.17   # Position of the nearest-neighbour correlation peak
# w = 0.1    # Width of the nearest-neighbour correlation peak

# ====================================================================================================

def Poisson_error_lowq_highq(q, Ierr_Poisson):
    mask = (q <= 0.1) | (q >= 4.1)
    Ierr_Poisson_improved = Ierr_Poisson.copy() * 1.7
    Ierr_Poisson_improved[mask] = Ierr_Poisson_improved[mask] * 2
    return Ierr_Poisson_improved

def op_error_classic(q, I):
    # mask = (q <= 0.1) | (q >= 4.1)
    Ierr_op = np.abs(I * 1/30) 
    # Ierr_op[mask] = np.abs(I[mask] * 1/10) 
    return Ierr_op

def Poisson_error_focus_minimum_highq(q, Ierr_Poisson):
    mask1 = (q <= 0.1)
    Ierr_op = Ierr_Poisson.copy()
    Ierr_op[mask1] = Ierr_op[mask1] * 2
    mask2 = ((q > 0.15) & (q < 0.4)) | (q >= 3)
    Ierr_op[mask2] = Ierr_op[mask2] * 0.1
    return Ierr_op

def Poisson_error_focus_lowq_and_minimum(q, Ierr_Poisson):
    Ierr_op = Ierr_Poisson.copy()
    mask2 = ((q > 0.15) & (q < 0.4)) | (q <= 0.1)
    Ierr_op[mask2] = Ierr_op[mask2] * 0.33
    return Ierr_op

def CSLB_op_error(q, I):
    I_err_op = np.copy(I) / 30
    op_err_mask = q > 0.6
    I_err_op[op_err_mask] = I[op_err_mask] / 15
    return I_err_op

def CSLB_weighted_op_error(q, I):
    I_err_op = np.copy(I) / 30
    op_err_mask1 = q > 0.6
    I_err_op[op_err_mask1] = I[op_err_mask1] / 15
    op_err_mask2 = q < 0.125
    I_err_op[op_err_mask2] = I[op_err_mask2] / 75
    return I_err_op

# ====================================================================================================

def get_chi2(sample, error_model, p0=None, fit_function="ULV_spheric_asym_C4"):
    block = sample.split("_")[0]
    block_path = blockpath(block)
    dir_names = [x.name for x in block_path.iterdir()]
    for dir_name in dir_names:
        if dir_name[:-13].endswith(sample):
            sub_dat_path = block_path / dir_name / "avg_and_sub" / sample_dict[sample]
            q, I, Ierr_Poisson = np.loadtxt(sub_dat_path, unpack=True)
    mask = (q >= 0.07) & (q <= 5.1)
    q    = q[mask]
    I    = I[mask]
    Ierr_Poisson = Ierr_Poisson[mask]
    if error_model == "Poisson_error_focus_minimum_highq":
        Ierr_op = Poisson_error_focus_minimum_highq(q, Ierr_Poisson)
    elif error_model == "op_error_classic":
        Ierr_op = op_error_classic(q, I)
    if p0 is None:
        p0 = best_fits[sample][fit_function][1]
    else:
        p0 = p0
    if fit_function == "ULV_spheric_asym_C4_lognormal":
        fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym_lognormal(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
    elif fit_function == "ULV_sph_asym_prot_layer":
        fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c4: fit.ULV_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, 0, 0, c4)
    elif fit_function == "ULV_spheric_asym_C4":
        fit_function = lambda q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale, c4: fit.ULV_spheric_asym(q, R,    sigma_R,   rho_h1,   rho_h2,   sigma_h1,  sigma_h2,  sigma_c,    z_h,   scale,      0, 0, c4)
    elif fit_function == "ULV_lognormal_sph_asym_prot_layer":
        fit_function = lambda q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c4: fit.ULV_lognormal_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, 0, 0, c4)
    I_sph_asym = fit_function(q, *p0)
    residuals     = I - I_sph_asym
    chi2 = np.sum((residuals / Ierr_op)**2) / (len(I) - len(p0) - 1)     
    return chi2

# ============================================================================================================
def linear(x, m, n):
    return m * x + n

def mean_and_sem(values):
    """
    Calculate ordinary arithmetic mean and SEM.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan, np.nan

    mean = np.mean(values)

    if values.size < 2:
        return mean, np.nan

    sem = np.std(values, ddof=1) / np.sqrt(values.size)

    return mean, sem

def calculate_linear_fit_statistics(x, y, yerr):
    """
    Perform linear fit and calculate:

        - slope and intercept
        - parameter uncertainties
        - R²
        - chi²
        - reduced chi²
        - F statistic
        - F-test p value
        - residual standard deviation

    The linear fit is weighted by yerr.
    MODIFIED: weighted R² is calculated in addition to the standard R². F-statistic is calculated using chi² ().
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(yerr)
        & (yerr > 0)
    )

    x = x[valid]
    y = y[valid]
    yerr = yerr[valid]

    # -------------------------------------------------------------------------
    # Linear fit
    # -------------------------------------------------------------------------

    popt, pcov = curve_fit(
        linear,
        x,
        y,
        sigma=yerr,
        p0=[0.0, 0.0],
        maxfev=20000,
    )

    perr = np.sqrt(np.diag(pcov))

    # -------------------------------------------------------------------------
    # Degrees of freedom
    # Two fitted parameters: slope + intercept
    # -------------------------------------------------------------------------

    dof = len(y) - len(popt)
    
    # -------------------------------------------------------------------------
    # Predictions and residuals
    # -------------------------------------------------------------------------

    y_pred = linear(x, *popt)
    residuals = y - y_pred

    # -------------------------------------------------------------------------
    # Chi² & R² Calculations (Properly Weighted)
    # -------------------------------------------------------------------------
    ymean = np.average(y, weights=1/yerr**2)

    # Chi-squared of the full model
    chi_squared = np.sum((residuals / yerr) ** 2)
    reduced_chi_squared = chi_squared / dof

    # Chi-squared of the flat baseline (intercept-only) model
    chi_squared_of_mean = np.sum(((y - ymean) / yerr) ** 2)

    # Weighted R²
    R2weighted = 1.0 - (chi_squared / chi_squared_of_mean)

    # Standard R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res / ss_tot

    # ---------------------------------------------------------------------
    # F-test
    # ---------------------------------------------------------------------
    # Numerator df = 1 (Slope parameter), Denominator df = dof
    f_stat = (chi_squared_of_mean - chi_squared) / (chi_squared / dof)

    p_value = f.sf(f_stat, 1, dof)

    # ---------------------------------------------------------------------
    # Residual standard deviation
    # ---------------------------------------------------------------------

    residual_std = np.sqrt(ss_res / dof)

    return {
        "popt": popt,
        "pcov": pcov,
        "perr": perr,

        "r_squared": R2,
        "r_squared_weighted": R2weighted,

        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,

        "f_stat": f_stat,
        "p_value": p_value,

        "residual_std": residual_std,

        "dof": dof,

        "x": x,
        "y": y,
        "yerr": yerr,
    }

def format_statistics_label(dataset_label, statistics):
    """
    Build the multiline legend entry containing the dataset name
    and all fit statistics.
    """

    r2 = statistics["r_squared_weighted"]
    red_chi2 = statistics["reduced_chi_squared"]
    f_stat = statistics["f_stat"]
    p_value = statistics["p_value"]

    return (
        f"{dataset_label}\n"
        f"R² = {r2:.3f}\n"
        f"$\chi^2_{{\mathrm{{red}}}} = {red_chi2:.3f}$\n"
        f"F = {f_stat:.3f}\n"
        f"p = {p_value:.3g}"
    )

# ===========================================================================================================

sample_dict = {
    "BII_S1": "BII_S1_15mM_LV4_TBS",
    "BII_S2": "BII_S2_15mM_LV4_30nm_PBS",
    "BII_S3": "BII_S3_1mM_LV4_PBS_FI",
    "BII_S4": "BII_S4_2mM_LV4_PBS_FII",
    "BII_S5": "BII_S5_3mM_LV4_PBS_FVII",
    "BII_S6": "BII_S6_6mM_LV4_PBS_FXIV",
    "BII_S7": "BII_S7_9mM_LV4_PBS_FI",
    "BII_S8": "BII_S8_12mM_LV4_PBS_FII",
    "BII_S9": "BII_S9_15mM_LV4_PBS_FVII",
    "BII_S10": "BII_S10_15mM_DOPS_PBS",
    "BII_S11": "BII_S11_15mM_DOPC_DOPS_PBS",

    "BIII_S1": "BIII_S1_75uM_aSynWT",
    "BIII_S2": "BIII_S2_100uM_aSynWT",
    "BIII_S3": "BIII_S3_100uM_aSynE61V",
    "BIII_S4": "BIII_S4_100uM_aSynA124T",
    "BIII_S5": "BIII_S5_100uM_aSynE46K",

    "BIV_S1": "BIV_S1_15mM_LV4_1uM_aSynWT",
    "BIV_S2": "BIV_S2_15mM_LV4_2uM_aSynWT",
    "BIV_S3": "BIV_S3_15mM_LV4_3uM_aSynWT",
    "BIV_S4": "BIV_S4_15mM_LV4_6uM_aSynWT",
    "BIV_S5": "BIV_S5_15mM_LV4_9uM_aSynWT",
    "BIV_S6": "BIV_S6_15mM_LV4_15uM_aSynWT",
    "BIV_S7": "BIV_S7_15mM_LV4_25uM_aSynWT",
    "BIV_S8": "BIV_S8_15mM_LV4_50uM_aSynWT",
    "BIV_S9": "BIV_S9_15mM_LV4_75uM_aSynWT",
    "BIV_S10": "BIV_S10_15mM_LV4_100uM_aSynWT",
    "BIV_S11": "BIV_S11_15mM_LV4_1uM_aSynE61V",
    "BIV_S12": "BIV_S12_15mM_LV4_2uM_aSynE61V",
    "BIV_S13": "BIV_S13_15mM_LV4_3uM_aSynE61V",
    "BIV_S14": "BIV_S14_15mM_LV4_6uM_aSynE61V",
    "BIV_S15": "BIV_S15_15mM_LV4_9uM_aSynE61V",
    "BIV_S16": "BIV_S16_15mM_LV4_15uM_aSynE61V",
    "BIV_S17": "BIV_S17_15mM_LV4_25uM_aSynE61V",
    "BIV_S18": "BIV_S18_15mM_LV4_50uM_aSynE61V",
    "BIV_S19": "BIV_S19_15mM_LV4_75uM_aSynE61V",
    "BIV_S20": "BIV_S20_15mM_LV4_100uM_aSynE61V",
    "BIV_S21": "BIV_S21_15mM_LV4_1uM_aSynA124T",
    "BIV_S22": "BIV_S22_15mM_LV4_2uM_aSynA124T",
    "BIV_S23": "BIV_S23_15mM_LV4_3uM_aSynA124T",
    "BIV_S24": "BIV_S24_15mM_LV4_6uM_aSynA124T",
    "BIV_S25": "BIV_S25_15mM_LV4_9uM_aSynA124T",
    "BIV_S26": "BIV_S26_15mM_LV4_15uM_aSynA124T",
    "BIV_S27": "BIV_S27_15mM_LV4_25uM_aSynA124T",
    "BIV_S28": "BIV_S28_15mM_LV4_50uM_aSynA124T",
    "BIV_S29": "BIV_S29_15mM_LV4_75uM_aSynA124T",
    "BIV_S30": "BIV_S30_15mM_LV4_100uM_aSynA124T",
    "BIV_S31": "BIV_S31_15mM_LV4_1uM_aSynE46K",
    "BIV_S32": "BIV_S32_15mM_LV4_2uM_aSynE46K",
    "BIV_S33": "BIV_S33_15mM_LV4_3uM_aSynE46K",
    "BIV_S34": "BIV_S34_15mM_LV4_6uM_aSynE46K",
    "BIV_S35": "BIV_S35_15mM_LV4_9uM_aSynE46K",
    "BIV_S36": "BIV_S36_15mM_LV4_15uM_aSynE46K",
    "BIV_S37": "BIV_S37_15mM_LV4_25uM_aSynE46K",
    "BIV_S38": "BIV_S38_15mM_LV4_50uM_aSynE46K",
    "BIV_S39": "BIV_S39_15mM_LV4_75uM_aSynE46K",
    "BIV_S40": "BIV_S40_15mM_LV4_100uM_aSynE46K",

    "BV_S1": "BV_S1_15mM_LV4_15uM_aSynG14R",
    "BV_S2": "BV_S2_15mM_LV4_25uM_aSynG14R",
    "BV_S3": "BV_S3_15mM_LV4_6uM_aSynA53T",
    "BV_S4": "BV_S4_15mM_LV4_9uM_aSynA53T",
    "BV_S5": "BV_S5_15mM_LV4_15uM_aSynA53T",
    "BV_S6": "BV_S6_15mM_LV4_25uM_aSynA53T",
    "BV_S7": "BV_S7_15mM_LV4_15uM_aSynA30P",
    "BV_S8": "BV_S8_15mM_LV4_25uM_aSynA30P",

    "BVI_S1": "BVI_S1_15mM_DOPC_GM1_6uM_aSynWT",
    "BVI_S2": "BVI_S2_15mM_DOPC_GM1_9uM_aSynWT",
    "BVI_S3": "BVI_S3_15mM_DOPC_GM1_15uM_aSynWT",
    "BVI_S4": "BVI_S4_15mM_DOPC_GM1_25uM_aSynWT",
    "BVI_S5": "BVI_S5_15mM_DOPC_GM1_50uM_aSynWT",
    "BVI_S6": "BVI_S6_15mM_DOPC_GM1_75uM_aSynWT",
    "BVI_S7": "BVI_S7_15mM_DOPC_GM1_100uM_aSynWT",
    "BVI_S8": "BVI_S8_15mM_DOPC_GM1_25uM_aSynE61V",
    "BVI_S9": "BVI_S9_15mM_DOPC_GM1_25uM_aSynA124T",
    "BVI_S10": "BVI_S10_15mM_DOPC_GM1_25uM_aSynE46K",
    "BVI_S11": "BVI_S11_15mM_POPG_6uM_aSynWT",
    "BVI_S12": "BVI_S12_15mM_POPG_9uM_aSynWT",
    "BVI_S13": "BVI_S13_15mM_POPG_15uM_aSynWT",
    "BVI_S14": "BVI_S14_15mM_POPG_25uM_aSynWT",
    "BVI_S15": "BVI_S15_15mM_POPG_50uM_aSynWT",
    "BVI_S16": "BVI_S16_15mM_POPG_75uM_aSynWT",
    "BVI_S17": "BVI_S17_15mM_POPG_100uM_aSynWT",
    "BVI_S18": "BVI_S18_15mM_POPG_25uM_aSynE61V",
    "BVI_S19": "BVI_S19_15mM_POPG_25uM_aSynA124T",
    "BVI_S20": "BVI_S20_15mM_POPG_25uM_aSynE46K",

    "BVII_S1": "BVII_S1_15mM_DOPS_75uM_aSynWT",
    "BVII_S2": "BVII_S2_15mM_DOPS_100uM_aSynWT",
    "BVII_S3": "BVII_S3_3mM_DOPS_100uM_aSynWT",
    "BVII_S4": "BVII_S4_15mM_DOPC_DOPS_75uM_aSynWT",
    "BVII_S5": "BVII_S5_15mM_DOPC_DOPS_100uM_aSynWT",
    "BVII_S6": "BVII_S6_3mM_DOPC_DOPS_100uM_aSynWT",
    "BVII_S7": "BVII_S7_3mM_LV4_100uM_aSynWT",
    "BVII_S8": "BVII_S8_15mM_DOPS_75uM_aSynE61V",
    "BVII_S9": "BVII_S9_3mM_DOPS_100uM_aSynE61V",
    "BVII_S10": "BVII_S10_15mM_DOPC_DOPS_75uM_aSynE61V",
    "BVII_S11": "BVII_S11_3mM_DOPC_DOPS_100uM_aSynE61V",
    "BVII_S12": "BVII_S12_3mM_LV4_100uM_aSynE61V",
    "BVII_S13": "BVII_S13_15mM_DOPS_75uM_aSynA124T",
    "BVII_S14": "BVII_S14_3mM_DOPS_100uM_aSynA124T",
    "BVII_S15": "BVII_S15_15mM_DOPC_DOPS_75uM_aSynA124T",
    "BVII_S16": "BVII_S16_3mM_DOPC_DOPS_100uM_aSynA124T",
    "BVII_S17": "BVII_S17_3mM_LV4_100uM_aSynA124T",
    "BVII_S18": "BVII_S18_15mM_DOPS_75uM_aSynE46K",
    "BVII_S19": "BVII_S19_3mM_DOPS_100uM_aSynE46K",
    "BVII_S20": "BVII_S20_15mM_DOPC_DOPS_75uM_aSynE46K",
    "BVII_S21": "BVII_S21_3mM_DOPC_DOPS_100uM_aSynE46K",
    "BVII_S22": "BVII_S22_3mM_LV4_100uM_aSynE46K",

    "BVIII_S1": "BVIII_S1_15mM_LV4_6uM_aSynWTpep",
    "BVIII_S2": "BVIII_S2_15mM_LV4_9uM_aSynWTpep",
    "BVIII_S3": "BVIII_S3_15mM_LV4_15uM_aSynWTpep",
    "BVIII_S4": "BVIII_S4_15mM_LV4_25uM_aSynWTpep",
    "BVIII_S5": "BVIII_S5_15mM_LV4_50uM_aSynWTpep",
    "BVIII_S6": "BVIII_S6_15mM_LV4_75uM_aSynWTpep",
    "BVIII_S7": "BVIII_S7_15mM_LV4_100uM_aSynWTpep",
    "BVIII_S8": "BVIII_S8_3mM_LV4_100uM_aSynWTpep",
    "BVIII_S9": "BVIII_S9_5mM_LV4_150uM_aSynWTpep",
    "BVIII_S11": "BVIII_S11_15mM_LV4_6uM_aSynG14Rpep",
    "BVIII_S12": "BVIII_S12_15mM_LV4_9uM_aSynG14Rpep",
    "BVIII_S13": "BVIII_S13_15mM_LV4_15uM_aSynG14Rpep",
    "BVIII_S14": "BVIII_S14_15mM_LV4_25uM_aSynG14Rpep",
    "BVIII_S15": "BVIII_S15_15mM_LV4_50uM_aSynG14Rpep",
    "BVIII_S16": "BVIII_S16_15mM_LV4_75uM_aSynG14Rpep",
    "BVIII_S17": "BVIII_S17_15mM_LV4_100uM_aSynG14Rpep",
    "BVIII_S18": "BVIII_S18_3mM_LV4_100uM_aSynG14Rpep",
    "BVIII_S19": "BVIII_S19_5mM_LV4_150uM_aSynG14Rpep",
    "BVIII_S20": "BVIII_S20_15mM_LV4_6uM_HBpep",
    "BVIII_S21": "BVIII_S21_15mM_LV4_9uM_HBpep",
    "BVIII_S22": "BVIII_S22_15mM_LV4_15uM_HBpep",
    "BVIII_S23": "BVIII_S23_15mM_LV4_25uM_HBpep",
    "BVIII_S24": "BVIII_S24_15mM_LV4_50uM_HBpep",
    "BVIII_S25": "BVIII_S25_15mM_LV4_75uM_HBpep",
    "BVIII_S26": "BVIII_S26_15mM_LV4_100uM_HBpep",
    "BVIII_S27": "BVIII_S27_3mM_LV4_100uM_HBpep",
    "BVIII_S28": "BVIII_S28_5mM_LV4_150uM_HBpep",
    "BVIII_S29": "BVIII_S29_5mM_LV4_250uM_HBpep",

    "BIX_S1": "BIX_S1_15mM_POPG",
    "BIX_S2": "BIX_S2_15mM_DOPC_GM1_80_20",
    "BIX_S3": "BIX_S3_15mM_LV4_1uM_aSynG14Rpep",
    "BIX_S4": "BIX_S4_15mM_LV4_2uM_aSynG14Rpep",
    "BIX_S5": "BIX_S5_15mM_LV4_3uM_aSynG14Rpep",
    "BIX_S6": "BIX_S6_15mM_LV4_6uM_aSynG14Rpep",
    "BIX_S7": "BIX_S7_15mM_LV4_9uM_aSynG14Rpep",
    "BIX_S8": "BIX_S8_15mM_LV4_15uM_aSynG14Rpep",
    "BIX_S9": "BIX_S9_15mM_LV4_25uM_aSynG14Rpep",
    "BIX_S10": "BIX_S10_15mM_DOPS_6uM_aSynWT",
    "BIX_S11": "BIX_S11_15mM_DOPS_9uM_aSynWT",
    "BIX_S12": "BIX_S12_15mM_DOPS_15uM_aSynWT",
    "BIX_S13": "BIX_S13_15mM_DOPS_25uM_aSynWT",
    "BIX_S14": "BIX_S14_15mM_DOPS_50uM_aSynWT",
    "BIX_S15": "BIX_S15_15mM_DOPC_DOPS_80_20_6uM_aSynWT",
    "BIX_S16": "BIX_S16_15mM_DOPC_DOPS_80_20_9uM_aSynWT",
    "BIX_S17": "BIX_S17_15mM_DOPC_DOPS_80_20_15uM_aSynWT",
    "BIX_S18": "BIX_S18_15mM_DOPC_DOPS_80_20_25uM_aSynWT",
    "BIX_S19": "BIX_S19_15mM_DOPC_DOPS_80_20_50uM_aSynWT",

    "BX_S1": "BX_S1_15mM_LV4",
    "BX_S4": "BX_S4_15mM_LV4_0p5μM_Syn1",
    "BX_S5": "BX_S5_15mM_LV4_1μM_Syn1",
    "BX_S6": "BX_S6_15mM_LV2",
    "BX_S9": "BX_S9_15mM_LV2_0p5μM_Syn1",
    "BX_S10": "BX_S10_15mM_LV2_1μM_Syn1",
    "BX_S11": "BX_S11_15mM_LV_Syx1a",
    "BX_S14": "BX_S14_15mM_LV_Syx1a_0p5μM_Syn1",
    "BX_S15": "BX_S15_15mM_LV_Syx1a_1μM_Syn1",
    "BX_S16": "BX_S16_15mM_LV_GramD1",
    "BX_S19": "BX_S19_15mM_LV_GramD1_0p5μM_Syn1",
    "BX_S20": "BX_S20_15mM_LV_GramD1_1μM_Syn1",
    "BX_S21": "BX_S21_15mM_LV_Syx1a_1to24_100nm",
    "BX_S24": "BX_S24_15mM_LV_Syx1a_1to24_100nm_0p5μM_Syn1",
    "BX_S25": "BX_S25_15mM_LV_Syx1a_1to24_100nm_1μM_Syn1",
    "BX_S26": "BX_S26_15mM_LV_GramD1_1to24_100nm",
    "BX_S29": "BX_S29_15mM_LV_GramD1_1to24_100nm_0p5μM_Syn1",
    "BX_S30": "BX_S30_15mM_LV_GramD1_1to24_100nm",
    "BX_S31": "BX_S31_15mM_LV4_4μM_Syn1",
    "BX_S32": "BX_S32_15mM_LV2_2μM_Syn1",
    "BX_S33": "BX_S33_15mM_LV4_2μM_Syn1",

    "BXII_S1": "BXII_S1_15mM_LV4_30nm",
    "BXII_S2": "BXII_S2_15mM_LV4_50nm", # WRONG BUFFER !!!
    "BXII_S3": "BXII_S3_15mM_LV4_100nm",
    "BXII_S4": "BXII_S4_15mM_LV4_200nm",
    "BXII_S6": "BXII_S6_15mM_DOPC",
    "BXII_S7": "BXII_S7_15mM_LV4_30nm_9uM_aSynWT",
    "BXII_S8": "BXII_S8_15mM_LV4_30nm_15uM_aSynWT",
    "BXII_S9": "BXII_S9_15mM_LV4_30nm_40uM_aSynWT",
    "BXII_S10": "BXII_S10_15mM_LV4_30nm_80uM_aSynWT",
    "BXII_S11": "BXII_S11_15mM_LV4_30nm_15uM_aSynE61V",
    "BXII_S12": "BXII_S12_15mM_LV4_100nm_9uM_aSynWT",
    "BXII_S13": "BXII_S13_15mM_LV4_100nm_15uM_aSynWT",
    "BXII_S14": "BXII_S14_15mM_LV4_100nm_40uM_aSynWT",
    "BXII_S15": "BXII_S15_15mM_LV4_100nm_80uM_aSynWT",
    "BXII_S16": "BXII_S16_15mM_LV4_100nm_15uM_aSynE61V",
    "BXII_S17": "BXII_S17_15mM_LV4_200nm_9uM_aSynWT",
    "BXII_S18": "BXII_S18_15mM_LV4_200nm_15uM_aSynWT",
    "BXII_S19": "BXII_S19_15mM_LV4_200nm_40uM_aSynWT",
    "BXII_S20": "BXII_S20_15mM_LV4_200nm_80uM_aSynWT",
    "BXII_S21": "BXII_S21_15mM_LV4_200nm_15uM_aSynE61V",
    "BXII_S27": "BXII_S27_15mM_DOPC_9uM_aSynWT",
    "BXII_S28": "BXII_S28_15mM_DOPC_15uM_aSynWT",
    "BXII_S29": "BXII_S29_15mM_DOPC_40uM_aSynWT",
    "BXII_S30": "BXII_S30_15mM_DOPC_80uM_aSynWT",
    "BXII_S31": "BXII_S31_15mM_DOPC_15uM_aSynE61V",
    "BXII_S32": "BXII_S32_15mM_LV4_aSynWT",
    "BXII_S33": "BXII_S33_15mM_LV4_aSynG14R",

    "BXIII_S1": "BXIII_S1_23nM_SVs",
    "BXIII_S2": "BXIII_S2_60%_CSLB",
    "BXIII_S3": "BXIII_S3_3uM_Syn1",
    "BXIII_S4": "BXIII_S4_15mM_LV4_4uM_Syn1",
    "BXIII_S5": "BXIII_S5_15mM_LV4_6uM_Syn1",
    "BXIII_S6": "BXIII_S6_15mM_LV4_6uM_Syn1_24uM_aSynWT",
    "BXIII_S11": "BXIII_S11_60%_CSLB_6uM_Syn1",
    "BXIII_S12": "BXIII_S12_60%_CSLB_4uM_Syn1",
    "BXIII_S13": "BXIII_S13_60%_CSLB_6uM_Syn1_12uM_aSynWT",
    "BXIII_S14": "BXIII_S14_60%_CSLB_6uM_Syn1_24uM_aSynWT",

    "BXIV_S1": "BXIV_S1_15mM_LV4_6uM_Syn1_24uM_ATP",
    "BXIV_S2": "BXIV_S2_23nM_SV_6uM_Syn1_24uM_ATP",
    "BXIV_S3": "BXIV_S3_60%_CSLB_6uM_Syn1_24uM_ATP"
}

nice_sample_names = {
    "BII_S1": "15 mM LVs (TBS)", # all other vesicles in PBS, also proteins
    "BII_S2": "15 mM LVs (30 nm)",
    "BII_S3": "1 mM LVs",
    "BII_S4": "2 mM LVs",
    "BII_S5": "3 mM LVs",
    "BII_S6": "6 mM LVs",
    "BII_S7": "9 mM LVs",
    "BII_S8": "12 mM LVs",
    "BII_S9": "15 mM LVs",
    "BII_S10": "15 mM DOPS",
    "BII_S11": "15 mM DOPC/DOPS (8/2)",

    "BIII_S1": "75 μM aSynWT",
    "BIII_S2": "100 μM aSynWT",
    "BIII_S3": "100 μM aSynE61V",
    "BIII_S4": "100 μM aSynA124T",
    "BIII_S5": "100 μM aSynE46K",
    
    "BIV_S1": "15 mM LVs + 1 μM aSynWT",
    "BIV_S2": "15 mM LVs + 2 μM aSynWT",
    "BIV_S3": "15 mM LVs + 3 μM aSynWT",
    "BIV_S4": "15 mM LVs + 6 μM aSynWT",
    "BIV_S5": "15 mM LVs + 9 μM aSynWT",
    "BIV_S6": "15 mM LVs + 15 μM aSynWT",
    "BIV_S7": "15 mM LVs + 25 μM aSynWT",
    "BIV_S8": "15 mM LVs + 50 μM aSynWT",
    "BIV_S9": "15 mM LVs + 75 μM aSynWT",
    "BIV_S10": "15 mM LVs + 100 μM aSynWT",
    "BIV_S11": "15 mM LVs + 1 μM aSynE61V",
    "BIV_S12": "15 mM LVs + 2 μM aSynE61V",
    "BIV_S13": "15 mM LVs + 3 μM aSynE61V",
    "BIV_S14": "15 mM LVs + 6 μM aSynE61V",
    "BIV_S15": "15 mM LVs + 9 μM aSynE61V",
    "BIV_S16": "15 mM LVs + 15 μM aSynE61V",
    "BIV_S17": "15 mM LVs + 25 μM aSynE61V",
    "BIV_S18": "15 mM LVs + 50 μM aSynE61V",
    "BIV_S19": "15 mM LVs + 75 μM aSynE61V",
    "BIV_S20": "15 mM LVs + 100 μM aSynE61V",
    "BIV_S21": "15 mM LVs + 1 μM aSynA124T",
    "BIV_S22": "15 mM LVs + 2 μM aSynA124T",
    "BIV_S23": "15 mM LVs + 3 μM aSynA124T",
    "BIV_S24": "15 mM LVs + 6 μM aSynA124T",
    "BIV_S25": "15 mM LVs + 9 μM aSynA124T",
    "BIV_S26": "15 mM LVs + 15 μM aSynA124T",
    "BIV_S27": "15 mM LVs + 25 μM aSynA124T",
    "BIV_S28": "15 mM LVs + 50 μM aSynA124T",
    "BIV_S29": "15 mM LVs + 75 μM aSynA124T",
    "BIV_S30": "15 mM LVs + 100 μM aSynA124T",
    "BIV_S31": "15 mM LVs + 1 μM aSynE46K",
    "BIV_S32": "15 mM LVs + 2 μM aSynE46K",
    "BIV_S33": "15 mM LVs + 3 μM aSynE46K",
    "BIV_S34": "15 mM LVs + 6 μM aSynE46K",
    "BIV_S35": "15 mM LVs + 9 μM aSynE46K",
    "BIV_S36": "15 mM LVs + 15 μM aSynE46K",
    "BIV_S37": "15 mM LVs + 25 μM aSynE46K",
    "BIV_S38": "15 mM LVs + 50 μM aSynE46K",
    "BIV_S39": "15 mM LVs + 75 μM aSynE46K",
    "BIV_S40": "15 mM LVs + 100 μM aSynE46K",

    "BV_S1": "15 mM LVs + 15 μM aSynG14R",
    "BV_S2": "15 mM LVs + 25 μM aSynG14R",
    "BV_S3": "15 mM LVs + 6 μM aSynA53T",
    "BV_S4": "15 mM LVs + 9 μM aSynA53T",
    "BV_S5": "15 mM LVs + 15 μM aSynA53T",
    "BV_S6": "15 mM LVs + 25 μM aSynA53T",
    "BV_S7": "15 mM LVs + 15 μM aSynA30P",
    "BV_S8": "15 mM LVs + 25 μM aSynA30P",

    "BVI_S1": "15 mM DOPC/GM1 (8/2) + 6 μM aSynWT",
    "BVI_S2": "15 mM DOPC/GM1 (8/2) + 9 μM aSynWT",
    "BVI_S3": "15 mM DOPC/GM1 (8/2) + 15 μM aSynWT",
    "BVI_S4": "15 mM DOPC/GM1 (8/2) + 25 μM aSynWT",
    "BVI_S5": "15 mM DOPC/GM1 (8/2) + 50 μM aSynWT",
    "BVI_S6": "15 mM DOPC/GM1 (8/2) + 75 μM aSynWT",
    "BVI_S7": "15 mM DOPC/GM1 (8/2) + 100 μM aSynWT",
    "BVI_S8": "15 mM DOPC/GM1 (8/2) + 25 μM aSynE61V",
    "BVI_S9": "15 mM DOPC/GM1 (8/2) + 25 μM aSynA124T",
    "BVI_S10": "15 mM DOPC/GM1 (8/2) + 25 μM aSynE46K",  
    "BVI_S11": "15 mM POPG + 6 μM aSynWT",
    "BVI_S12": "15 mM POPG + 9 μM aSynWT",
    "BVI_S13": "15 mM POPG + 15 μM aSynWT",
    "BVI_S14": "15 mM POPG + 25 μM aSynWT",
    "BVI_S15": "15 mM POPG + 50 μM aSynWT",
    "BVI_S16": "15 mM POPG + 75 μM aSynWT",
    "BVI_S17": "15 mM POPG + 100 μM aSynWT",
    "BVI_S18": "15 mM POPG + 25 μM aSynE61V",
    "BVI_S19": "15 mM POPG + 25 μM aSynA124T",
    "BVI_S20": "15 mM POPG + 25 μM aSynE46K",

    "BVII_S1": "15 mM DOPS + 75 μM aSynWT",
    "BVII_S2": "15 mM DOPS + 100 μM aSynWT",
    "BVII_S3": "3 mM DOPS + 100 μM aSynWT",
    "BVII_S4": "15 mM DOPC/DOPS (8/2) + 75 μM aSynWT",
    "BVII_S5": "15 mM DOPC/DOPS (8/2) + 100 μM aSynWT",
    "BVII_S6": "3 mM DOPC/DOPS (8/2) + 100 μM aSynWT",
    "BVII_S7": "3 mM LVs + 100 μM aSynWT",
    "BVII_S8": "15 mM DOPS + 75 μM aSynE61V",
    "BVII_S9": "3 mM DOPS + 100 μM aSynE61V",
    "BVII_S10": "15 mM DOPC/DOPS (8/2) + 75 μM aSynE61V",
    "BVII_S11": "3 mM DOPC/DOPS (8/2) + 100 μM aSynE61V",
    "BVII_S12": "3 mM LVs + 100 μM aSynE61V",
    "BVII_S13": "15 mM DOPS + 75 μM aSynA124T",
    "BVII_S14": "3 mM DOPS + 100 μM aSynA124T",
    "BVII_S15": "15 mM DOPC/DOPS (8/2) + 75 μM aSynA124T",
    "BVII_S16": "3 mM DOPC/DOPS (8/2) + 100 μM aSynA124T",
    "BVII_S17": "3 mM LVs + 100 μM aSynA124T",
    "BVII_S18": "15 mM DOPS + 75 μM aSynE46K",
    "BVII_S19": "3 mM DOPS + 100 μM aSynE46K",
    "BVII_S20": "15 mM DOPC/DOPS (8/2) + 75 μM aSynE46K",
    "BVII_S21": "3 mM DOPC/DOPS (8/2) + 100 μM aSynE46K",
    "BVII_S22": "3 mM LVs + 100 μM aSynE46K",

    "BVIII_S1": "15 mM LVs + 6 μM aSynWT (1-25)",
    "BVIII_S2": "15 mM LVs + 9 μM aSynWT (1-25)",
    "BVIII_S3": "15 mM LVs + 15 μM aSynWT (1-25)",
    "BVIII_S4": "15 mM LVs + 25 μM aSynWT (1-25)",
    "BVIII_S5": "15 mM LVs + 50 μM aSynWT (1-25)",
    "BVIII_S6": "15 mM LVs + 75 μM aSynWT (1-25)",
    "BVIII_S7": "15 mM LVs + 100 μM aSynWT (1-25)",
    "BVIII_S8": "3 mM LVs + 100 μM aSynWT (1-25)",
    "BVIII_S9": "5 mM LVs + 150 μM aSynWT (1-25)",
    "BVIII_S11": "15 mM LVs + 6 μM aSynG14R (1-25)",
    "BVIII_S12": "15 mM LVs + 9 μM aSynG14R (1-25)",
    "BVIII_S13": "15 mM LVs + 15 μM aSynG14R (1-25)",
    "BVIII_S14": "15 mM LVs + 25 μM aSynG14R (1-25)",
    "BVIII_S15": "15 mM LVs + 50 μM aSynG14R (1-25)",
    "BVIII_S16": "15 mM LVs + 75 μM aSynG14R (1-25)",
    "BVIII_S17": "15 mM LVs + 100 μM aSynG14R (1-25)",
    "BVIII_S18": "3 mM LVs + 100 μM aSynG14R (1-25)",
    "BVIII_S19": "5 mM LVs + 150 μM aSynG14R (1-25)",    
    "BVIII_S20": "15 mM LVs + 6 μM HBpep",
    "BVIII_S21": "15 mM LVs + 9 μM HBpep",
    "BVIII_S22": "15 mM LVs + 15 μM HBpep",
    "BVIII_S23": "15 mM LVs + 25 μM HBpep",
    "BVIII_S24": "15 mM LVs + 50 μM HBpep",
    "BVIII_S25": "15 mM LVs + 75 μM HBpep",
    "BVIII_S26": "15 mM LVs + 100 μM HBpep",
    "BVIII_S27": "3 mM LVs + 100 μM HBpep",
    "BVIII_S28": "5 mM LVs + 150 μM HBpep",
    "BVIII_S29": "5 mM LVs + 250 μM HBpep",

    "BIX_S1": "15 mM POPG",
    "BIX_S2": "15 mM DOPC/GM1 (8/2)",
    
    "BIX_S3": "15 mM LVs + 1 μM aSynG14R (1-25)",
    "BIX_S4": "15 mM LVs + 2 μM aSynG14R (1-25)",
    "BIX_S5": "15 mM LVs + 3 μM aSynG14R (1-25)",
    "BIX_S6": "15 mM LVs + 6 μM aSynG14R (1-25)",
    "BIX_S7": "15 mM LVs + 9 μM aSynG14R (1-25)",
    "BIX_S8": "15 mM LVs + 15 μM aSynG14R (1-25)",
    "BIX_S9": "15 mM LVs + 25 μM aSynG14R (1-25)",  
    "BIX_S10": "15 mM DOPS + 6 μM aSynWT",
    "BIX_S11": "15 mM DOPS + 9 μM aSynWT",
    "BIX_S12": "15 mM DOPS + 15 μM aSynWT",
    "BIX_S13": "15 mM DOPS + 25 μM aSynWT",
    "BIX_S14": "15 mM DOPS + 50 μM aSynWT",
    "BIX_S15": "15 mM DOPC/DOPS (8/2) + 6 μM aSynWT",
    "BIX_S16": "15 mM DOPC/DOPS  (8/2) + 9 μM aSynWT",
    "BIX_S17": "15 mM DOPC/DOPS (8/2) + 15 μM aSynWT",
    "BIX_S18": "15 mM DOPC/DOPS (8/2) + 25 μM aSynWT",
    "BIX_S19": "15 mM DOPC/DOPS (8/2) + 50 μM aSynWT",

    "BX_S1": "15 mM LV4",
    "BX_S4": "15 mM LV4 + 0.5 μM Syn1",
    "BX_S5": "15 mM LV4 + 1 μM Syn1",
    "BX_S6": "15 mM LV2",
    "BX_S9": "15 mM LV2 + 0.5 μM Syn1",
    "BX_S10": "15 mM LV2 + 1 μM Syn1",
    "BX_S11": "15 mM LV_Syx1a",
    "BX_S14": "15 mM LV_Syx1a + 0.5 μM Syn1",
    "BX_S15": "15 mM LV_Syx1a + 1 μM Syn1",
    "BX_S16": "15 mM LV_GramD1",
    "BX_S19": "15 mM LV_GramD1 + 0.5 μM Syn1",
    "BX_S20": "15 mM LV_GramD1 + 1 μM Syn1",
    "BX_S21": "15 mM LV_Syx1a_1to24 (100 nm)",
    "BX_S24": "15 mM LV_Syx1a_1to24 (100nm) + 0.5 μM Syn1",
    "BX_S25": "15 mM LV_Syx1a_1to24 (100nm) + 1 μM Syn1",
    "BX_S26": "15 mM LV_GramD1_1to24 (100nm)",
    "BX_S29": "15 mM LV_GramD1_1to24 (100nm) + 0.5 μM Syn1",
    "BX_S30": "15 mM LV_GramD1_1to24 (100nm)",
    "BX_S31": "15 mM LV4 + 4 μM Syn1",
    "BX_S32": "15 mM LV2  + 2 μM Syn1",
    "BX_S33": "15 mM LV4 + 2 μM Syn1",

    "BXII_S1": "15 mM LV4 (30nm)",
    "BXII_S2": "15 mM LV4 (50nm)", # WRONG BUFFER !!!
    "BXII_S3": "15 mM LV4 (100nm)",
    "BXII_S4": "15 mM LV4 (200nm)",
    "BXII_S5": "15 mM DMPS",
    "BXII_S6": "15 mM DOPC",
    "BXII_S7": "15 mM LV4 (30nm) + 9 μM aSyn WT",
    "BXII_S8": "15 mM LV4 (30nm) + 15 μM aSyn WT",
    "BXII_S9": "15 mM LV4 (30nm) + 40 μM aSyn WT",
    "BXII_S10": "15 mM LV4 (30nm) + 80 μM aSyn WT",
    "BXII_S11": "15 mM LV4 (30nm) + 15 μM aSyn E61V",
    "BXII_S12": "15 mM LV4 (100nm) + 9 μM aSyn WT",
    "BXII_S13": "15 mM LV4 (100nm) + 15 μM aSyn WT",
    "BXII_S14": "15 mM LV4 (100nm) + 40 μM aSyn WT",
    "BXII_S15": "15 mM LV4 (100nm) + 80 μM aSyn WT",
    "BXII_S16": "15 mM LV4 (100nm) + 15 μM aSyn E61V",
    "BXII_S17": "15 mM LV4 (200nm) + 9 μM aSyn WT",
    "BXII_S18": "15 mM LV4 (200nm) + 15 μM aSyn WT",
    "BXII_S19": "15 mM LV4 (200nm) + 40 μM aSyn WT",
    "BXII_S20": "15 mM LV4 (200nm) + 80 μM aSyn WT",
    "BXII_S21": "15 mM LV4 (200nm) + 15 μM aSyn E61V",
    "BXII_S22": "15 mM DMPS + 9 μM aSyn WT",
    "BXII_S23": "15 mM DMPS + 15 μM aSyn WT",
    "BXII_S24": "15 mM DMPS + 40 μM aSyn WT",
    "BXII_S25": "15 mM DMPS + 80 μM aSyn WT",
    "BXII_S26": "15 mM DMPS + 15 μM aSyn E61V",
    "BXII_S27": "15 mM DOPC + 9 μM aSyn WT",
    "BXII_S28": "15 mM DOPC + 15 μM aSyn WT",
    "BXII_S29": "15 mM DOPC + 40 μM aSyn WT",
    "BXII_S30": "15 mM DOPC + 80 μM aSyn WT",
    "BXII_S31": "15 mM DOPC + 15 μM aSyn E61V",
    "BXII_S32": "15 mM LV4_aSynWT",
    "BXII_S33": "15 mM LV4_aSynG14R",

    "BXIII_S1": "23nM SVs",
    "BXIII_S2": "CSLBs",
    "BXIII_S3": "3 μM Syn1",
    "BXIII_S4": "15mM LV4 + 4μM Syn1",
    "BXIII_S5": "15mM LV4 + 6μM Syn1",
    "BXIII_S6": "15mM LV4 + 6μM Syn1 + 24μM aSynWT",
    "BXIII_S11": "CSLB + 6μM Syn1",
    "BXIII_S12": "CSLB + 4μM Syn1",
    "BXIII_S13": "CSLB + 6μM Syn1 + 12μM aSynWT",
    "BXIII_S14": "CSLB + 6μM Syn1 + 24μM aSynWT",

    "BXIV_S1": "15mM LV4 + 6μM Syn1 + 24μM ATP",
    "BXIV_S2": "23nM SVs + 6μM Syn1 + 24μM ATP",
    "BXIV_S3": "CSLB + 6μM Syn1 + 24 μM ATP"
}

concentrations = {
    "BII_S1": 0, # all other vesicles in PBS, also proteins
    "BII_S2": 0,
    "BII_S9": 0,
    "BII_S10": 0,
    "BII_S11": 0,

    "BIV_S1": 1.0,
    "BIV_S2": 2.0,
    "BIV_S3": 3.0,
    "BIV_S4": 6.0,
    "BIV_S5": 9.0,
    "BIV_S6": 15.0,
    "BIV_S7": 25,
    "BIV_S8": 50,
    "BIV_S9": 75,
    "BIV_S10": 100,

    "BIV_S11": 1.0,
    "BIV_S12": 2.0,
    "BIV_S13": 3.0,
    "BIV_S14": 6.0,
    "BIV_S15": 9.0,
    "BIV_S16": 15.0,
    "BIV_S17": 25,
    "BIV_S18": 50,
    "BIV_S19": 75,
    "BIV_S20": 100,

    "BIV_S21": 1.0,
    "BIV_S22": 2.0,
    "BIV_S23": 3.0,
    "BIV_S24": 6.0,
    "BIV_S25": 9.0,
    "BIV_S26": 15.0,
    "BIV_S27": 25,
    "BIV_S28": 50,
    "BIV_S29": 75,
    "BIV_S30": 100,

    "BIV_S31": 1.0,
    "BIV_S32": 2.0,
    "BIV_S33": 3.0,
    "BIV_S34": 6.0,
    "BIV_S35": 9.0,
    "BIV_S36": 15.0,
    "BIV_S37": 25,
    "BIV_S38": 50,
    "BIV_S39": 75,
    "BIV_S40": 100,

    "BVI_S1": 6.0,
    "BVI_S2": 9.0,
    "BVI_S3": 15.0,
    "BVI_S4": 25.0,
    "BVI_S5": 50.0,
    "BVI_S6": 75.0,
    "BVI_S7": 100,
    "BVI_S8": 25,
    "BVI_S9": 25,
    "BVI_S10": 25,

    "BVI_S11": 6.0,
    "BVI_S12": 9.0,
    "BVI_S13": 15.0,
    "BVI_S14": 25.0,
    "BVI_S15": 50.0,
    "BVI_S16": 75.0,
    "BVI_S17": 100,
    "BVI_S18": 25,
    "BVI_S19": 25,
    "BVI_S20": 25,

    "BVII_S1": 75,
    "BVII_S2": 100,
    "BVII_S3": 100,
    "BVII_S4": 75,
    "BVII_S5": 100,
    "BVII_S6": 100,
    "BVII_S7": 100,
    "BVII_S8": 75,
    "BVII_S9": 100,
    "BVII_S10": 75,
    "BVII_S11": 100,
    "BVII_S12": 100,
    "BVII_S13": 75,
    "BVII_S14": 100,
    "BVII_S15": 75,
    "BVII_S16": 100,
    "BVII_S17": 100,
    "BVII_S18": 75,
    "BVII_S19": 100,
    "BVII_S20": 75,
    "BVII_S21": 100,
    "BVII_S22": 100,

    "BIX_S1": 0,
    "BIX_S2": 0,
    "BIX_S10": 6,
    "BIX_S11": 9,
    "BIX_S12": 15,
    "BIX_S13": 25,
    "BIX_S14": 50,
    "BIX_S15": 6,
    "BIX_S16": 9,
    "BIX_S17": 15,
    "BIX_S18": 25,
    "BIX_S19": 50,

    # Syn1 concentrations
    "BX_S1": 0,
    "BX_S4": 0.5,
    "BX_S5": 1,
    "BX_S6": 0,
    "BX_S9": 0.5,
    "BX_S10": 1,
    "BX_S11": 0,
    "BX_S14": 0.5,
    "BX_S15": 1,
    "BX_S16": 0,
    "BX_S19": 0.5,
    "BX_S20": 1,
    "BX_S21": 0,
    "BX_S24": 0.5,
    "BX_S25": 1,
    "BX_S26": 0,
    "BX_S29": 0.5,
    "BX_S30": 0,
    "BX_S31": 4,
    "BX_S32": 2,
    "BX_S33": 2,

    "BXII_S1": 0,
    "BXII_S2": 0, # WRONG BUFFER !!!
    "BXII_S3": 0,
    "BXII_S4": 0,
    "BXII_S5": 0,
    "BXII_S6": 0,
    "BXII_S7": 9,
    "BXII_S8": 15,
    "BXII_S9": 40,
    "BXII_S10": 80,
    "BXII_S11": 15,
    "BXII_S12": 9,
    "BXII_S13": 15,
    "BXII_S14": 40,
    "BXII_S15": 80,
    "BXII_S16": 15,
    "BXII_S17": 9,
    "BXII_S18": 15,
    "BXII_S19": 40,
    "BXII_S20": 80,
    "BXII_S21": 15,
    "BXII_S22": 9,
    "BXII_S23": 15,
    "BXII_S24": 40,
    "BXII_S25": 80,
    "BXII_S26": 15,
    "BXII_S27": 9,
    "BXII_S28": 15,
    "BXII_S29": 40,
    "BXII_S30": 80,
    "BXII_S31": 15,
    # "BXII_S32": "15 mM LV4_aSynWT",
    # "BXII_S33": "15 mM LV4_aSynG14R",

    "BXIII_S1": 0,
    "BXIII_S2": 0,
    "BXIII_S3": 3,
    "BXIII_S4": 4,
    "BXIII_S5": 6,
    # "BXIII_S6": "15mM LV4 + 6μM Syn1 + 24μM aSynWT",
    # "BXIII_S11": "CSLB + 6μM Syn1",
    # "BXIII_S12": "CSLB + 4μM Syn1",
    # "BXIII_S13": "CSLB + 6μM Syn1 + 12μM aSynWT",
    # "BXIII_S14": "CSLB + 6μM Syn1 + 24μM aSynWT",

    # "BXIV_S1": "15mM LV4 + 6μM Syn1 + 24μM ATP",
    # "BXIV_S2": "23nM SVs + 6μM Syn1 + 24μM ATP",
    # "BXIV_S3": "CSLB + 6μM Syn1 + 24 μM ATP"
}

best_fits = {
    'BII_S2': {
        'ULV_spheric_asym_C4': {
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [8, 0.6705, 1.356, 1.009, 0.292, 0.3374, 0.6438, 1.921, 5.304e-07, 0.0005005],
            2: [9.001, 0.6196, 1.298, 1.044, 0.3013, 0.3333, 0.6462, 1.921, 4.709e-07, 0.0005032],
            3: [9, 0.5787, 1.091, 1.608, 0.3986, 0.1919, 0.6419, 1.928, 5.187e-07, 0.000445],
        },
        'ULV_ellipsoid_asym_C4': {
        }
    },
    'BII_S4': {
        'ULV_spheric_asym_C4': {
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4995, 1.322, 1.159, 0.3933, 0.4187, 0.8871, 1.831, 6.142e-08, 1.038e-32],
            2: [11.69, 0.3445, 1.467, 1.03, 0.4122, 0.3952, 0.8777, 1.833, 4.768e-08, 3.347e-39],
        },
        'ULV_ellipsoid_asym_C4': {
        }
    },
    'BII_S9': {
        "flat_asym_C4_q0p7": {
            #  sigma_h1, sigma_h2, sigma_tail, rho_h1,  rho_h2,   eps_head,       scale,     c4
            1: [0.2481,    0.3061,     0.6217,      1,   1.463,      1.908,      0.0488, 0.000501]
        },
        'ULV_spheric_asym_C4': {
            1: [16.09, 2.936, 1.383, 0.9795, 0.3527, 0.3049, 0.6648, 1.907, 0.0001727, 0.0005608],
            2: [8.864, 4.584, 1.333, 1.024, 0.3194, 0.3458, 0.6705, 1.906, 0.0004622, 0.0005589],
        },
        "ULV_spheric_asym_C4_lognormal": {
            # Poisson_error_focus_minimum_highq
            1: [9.25, 0.4552, 1.261, 1.071, 0.3214, 0.3518, 0.6763, 1.906, 7.089e-07, 0.0005629],
            # op_error_classic
            2: [9.58, 0.6004, 3, 1.061, 0.2093, 0.6, 1.158, 1.78, 4.645e-07, 6.903e-05],
            # Poisson_error_lowq_highq
            3: [],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.89, 2.825, 0.8596, 1.468e-08, 1.473, 1.079, 0.3535, 0.2949, 0.716, 1.989, 3.957e-07, 0.0004725],
            # 2: [13.19, 4.587, 1.081, 0.4, 2.9, 2.509, 0.1307, 0.1, 0.5185, 1.87, 5.487e-07, 0.0002264],
            # 3: [16.16, 3.131, 1.133, 0.2141, 1.381, 1.035, 0.3428, 0.2975, 0.657, 1.807, 3.14e-07, 0.000499],
            # 4: [16.26, 3.134, 1.09, 0.2514, 1.388, 1.073, 0.343, 0.2949, 0.6669, 1.821, 3.19e-07, 0.0004654]
        }
    },
    'BII_S10': {
        'ULV_spheric_asym_C4': {
            1: [21.3, 9.471, 1.896, 1.401, 0.3401, 0.3958, 0.8136, 1.931, 9.347e-05, 0.0006512],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.57, 0.6083, 2.561, 1.179, 0.2433, 0.4465, 0.7663, 1.937, 3.934e-07, 0.0005002],
        },
    },
    'BII_S11': {
        'ULV_spheric_asym_C4': { # 15 mM DOPC/DOPS (8/2)
            1: [16.4, 2.966, 1.594, 0.9678, 0.3711, 0.3601, 0.8142, 1.78, 0.0001319, 0.0004316],
            # 2: [9, 4.168, 1.29, 1.279, 0.3961, 0.3705, 0.8757, 1.768, 0.0003619, 0.0004293],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [15.88, 0.1773, 1.673, 0.9529, 0.3631, 0.3883, 0.8495, 1.769, 2.692e-07, 0.0004229],
        },
    },
    'BIV_S1': {
        'ULV_spheric_asym_C4': {
            1: [16.46, 2.746, 1.365, 0.949, 0.35, 0.3118, 0.6541, 1.909, 0.0001544, 0.0005382],
            2: [9.213, 4.424, 1.3, 1.004, 0.3213, 0.345, 0.6579, 1.908, 0.0004128, 0.0005378],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.216, 0.4407, 1.241, 1.036, 0.3197, 0.3477, 0.6543, 1.911, 6.953e-07, 0.0005402],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.26, 2.946, 0.8713, 0.0916, 1.465, 1.049, 0.382, 0.3424, 0.7931, 1.941, 3.333e-07, 0.0003789],
            # 2: [16.27, 2.865, 0.7452, 0.3678, 1.366, 0.9596, 0.34, 0.3265, 0.6587, 1.929, 4.687e-07, 0.0004716],
            # 3: [16.34, 2.944, 0.732, 0.463, 1.271, 1.049, 0.34, 0.3135, 0.6394, 1.886, 4.841e-07, 0.0002168]
        }
    },
    'BIV_S2': {
        'ULV_spheric_asym_C4': {
            1: [16.35, 2.843, 1.353, 0.9624, 0.3525, 0.3089, 0.6547, 1.908, 0.0001566, 0.0004852],
            2: [9.236, 4.599, 1.31, 1.001, 0.3205, 0.3459, 0.6585, 1.907, 0.0004053, 0.0004842],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1:  [10.18, 4.776, 0.9143, 0.2056, 1.884, 1.166, 0.3172, 0.3419, 0.8674, 1.858, 6.867e-07, 6.052e-05],
            # 2: [10.04, 4.772, 0.9709, 0.2317, 1.377, 1.108, 0.3204, 0.3421, 0.7087, 1.879, 6.977e-07, 0.0004477],
        }
    },
    'BIV_S3': {
        'ULV_spheric_asym_C4': {
            1: [16.34, 2.899, 1.399, 0.9106, 0.3549, 0.3456, 0.6896, 1.892, 0.0001451, 0.0004615],
            2: [8.973, 4.663, 1.2, 1.2, 0.3624, 0.2917, 0.6757, 1.902, 0.0003923, 0.0004199],
            3: [15.78, 2.881, 1.2, 1.2, 0.4048, 0.2536, 0.6741, 1.896, 0.0001594, 0.0004388],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [10.26, 4.722, 0.9766, 0.1896, 1.294, 1.125, 0.3465, 0.3501, 0.7296, 1.878, 6.305e-07, 0.0004352]
        }
    },
    'BIV_S4': {
        'ULV_spheric_asym_C4': {
            1: [16.56, 3.14, 1.541, 0.8599, 0.3301, 0.3726, 0.705, 1.887, 0.0001312, 0.0004451],
            2: [9.565, 4.81, 1.197, 1.127, 0.3676, 0.3335, 0.7069, 1.886, 0.0003267, 0.0004478],
            3: [15.95, 3.259, 1.235, 1.2, 0.4041, 0.2597, 0.692, 1.89, 0.0001441, 0.0004347],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [15.47, 0.1933, 1.432, 0.9518, 0.3468, 0.3275, 0.6884, 1.893, 2.967e-07, 0.000445],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.48, 3.33, 0.8856, 0.2918, 1.543, 0.9217, 0.3228, 0.3626, 0.7047, 1.893, 3.148e-07, 0.0004243]
        }
    },
    'BIV_S5': {
        'ULV_spheric_asym_C4': {
            1: [16.05, 3.551, 1.408, 0.9312, 0.3456, 0.3181, 0.6606, 1.901, 0.0001454, 0.0004653],
            2: [8.406, 5.15, 1.295, 1.034, 0.3208, 0.3401, 0.6595, 1.9, 0.0004063, 0.000462],
            3: [15.09, 3.757, 1.242, 1.2, 0.3874, 0.2418, 0.6526, 1.902, 0.0001655, 0.0004467],
            4: [11.99, 5.111, 1.647, 1, 0.4067, 0.4343, 0.9656, 1.813, 0.0001926, 0.000456],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.5003, 1.185, 1.114, 0.3301, 0.3413, 0.6701, 1.897, 5.873e-07, 0.0004613],
            2: [14.23, 0.2601, 1.446, 0.9063, 0.3368, 0.3295, 0.6659, 1.898, 3.408e-07, 0.000461],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.01, 3.748, 0.9081, 0.2396, 1.508, 1.018, 0.3356, 0.3169, 0.6991, 1.904, 3.16e-07, 0.0004296],
            # 1: [16.25, 3.751, 1.14, 0.2063, 1.405, 0.982, 0.3356, 0.3081, 0.6468, 1.801, 2.593e-07, 0.0004446]
        }
    },
    'BIV_S6': {
        'ULV_spheric_asym_C4': {
            1: [16.23, 3.431, 1.547, 0.8343, 0.3279, 0.3607, 0.6795, 1.891, 0.0001441, 0.0004692],
            2: [8.464, 4.847, 1.119, 1.196, 0.3733, 0.3092, 0.6778, 1.891, 0.0004192, 0.000459],
            3: [15.56, 3.534, 1.251, 1.2, 0.4, 0.2446, 0.6702, 1.892, 0.0001594, 0.0004561],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.485, 0.4691, 1.053, 1.254, 0.3753, 0.3175, 0.6879, 1.892, 5.659e-07, 0.000456],
            2: [14.95, 0.2426, 1.633, 0.809, 0.3078, 0.3714, 0.6767, 1.89, 3.184e-07, 0.0004524],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.03, 3.626, 0.9201, 0.2254, 1.635, 0.9342, 0.323, 0.3509, 0.7203, 1.891, 3.143e-07, 0.0004281]
        }
    },
    'BIV_S7': {
        'ULV_spheric_asym_C4': {
            1: [16.08, 4.223, 1.627, 0.8058, 0.3221, 0.3626, 0.6807, 1.888, 0.0001421, 0.0004869],
            2: [14.32, 2.608, 1.4, 1.008, 0.3572, 0.2629, 0.6511, 1.892, 0.0001983, 0.0004913],
            3: [7.336, 5.501, 1.2, 1.2, 0.3345, 0.3156, 0.6709, 1.893, 0.0004699, 0.0004715],
            4: [15.11, 4.424, 1.301, 1.2, 0.3984, 0.2372, 0.6708, 1.89, 0.0001624, 0.0004818],
            5: [13.41, 5.053, 1.434, 1, 0.4024, 0.3236, 0.7597, 1.858, 0.0001804, 0.0004952],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.369, 0.4965, 1.043, 1.256, 0.3543, 0.3226, 0.6694, 1.895, 5.574e-07, 0.0004888],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.03, 4.255, 0.7812, 0.4446, 1.536, 0.8167, 0.3121, 0.3526, 0.6357, 1.861, 4.324e-07, 0.0004522],
            # 1: [16.27, 4.376, 0.9152, 0.338, 1.539, 0.8157, 0.3121, 0.3526, 0.6338, 1.859, 3.516e-07, 0.0003248]
        }
    },
    'BIV_S8': {
        "flat_asym_C4_q0p7": {
            1: [0.2442, 0.3632, 0.6252, 1, 1.684, 1.89, 0.04044, 0.0006435],
        },
        'ULV_spheric_asym_C4': {
            1: [16.15, 5.084, 1.641, 0.8074, 0.3176, 0.3193, 0.6389, 1.9, 0.0001657, 0.0006746],
            2: [14.88, 5.343, 1.4, 1.102, 0.3729, 0.2315, 0.638, 1.897, 0.0001926, 0.0006815],
            3: [14, 8, 1, 1.326, 0.3183, 0.3152, 0.638, 1.9, 0.0001834, 0.0006723],
            4: [21.33, 8.304, 1.418, 1, 0.3537, 0.2635, 0.6392, 1.898, 9.152e-05, 0.0006849],
            5: [16.76, 8.549, 1.201, 1.2, 0.2658, 0.3596, 0.6522, 1.894, 0.0001331, 0.0006806],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.77, 0.5059, 1.066, 1.277, 0.3042, 0.3295, 0.6418, 1.898, 4.931e-07, 0.0006817],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.35, 5.284, 1.098, 0.1713, 1.605, 0.8125, 0.3115, 0.3098, 0.6131, 1.826, 3.133e-07, 0.0006509],
            # 2: [16.34, 5.264, 0.9524, 0.277, 1.617, 0.8386, 0.3114, 0.3099, 0.6227, 1.873, 3.687e-07, 0.0005159],
            # 3: [16.32, 5.281, 0.8406, 0.3059, 1.657, 0.8572, 0.31, 0.3097, 0.6366, 1.918, 4.142e-07, 0.0006252],
            # 1: [16.35, 5.284, 1.15, 0.1457, 1.636, 0.8238, 0.3076, 0.3093, 0.6191, 1.8, 2.949e-07, 0.000652],
        }
    },
    'BIV_S9': {
        "flat_asym_C4_q0p7": {
            1: [0.291, 0.4027, 0.7033, 1, 2.036, 1.857, 0.02832, 0.0006657]
        },
        'ULV_spheric_asym_C4': {
            1: [21.07, 7.182, 1.729, 0.8, 0.3169, 0.3059, 0.6517, 1.887, 8.572e-05, 0.000683],
            2: [16.66, 6, 1.4, 1.4, 0.4042, 0.1748, 0.6606, 1.88, 0.0001376, 0.000698],
            3: [8.229, 6, 1.2, 1.263, 0.2657, 0.3434, 0.6508, 1.886, 0.000411, 0.0006962],
            4: [17.06, 6, 1.002, 1.4, 0.2901, 0.3256, 0.6486, 1.886, 0.0001311, 0.0006917],
            5: [16.42, 6, 1.6, 0.9298, 0.3521, 0.2605, 0.6547, 1.884, 0.0001395, 0.0007035],
            6: [11.23, 8, 1.4, 1.251, 0.4092, 0.2, 0.6729, 1.874, 0.0002256, 0.0007079],
            7: [15.57, 6.472, 1.897, 0.7222, 0.295, 0.331, 0.647, 1.889, 0.0001494, 0.0006637],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.33, 0.5352, 0.8798, 1.571, 0.3423, 0.2963, 0.6616, 1.888, 3.703e-07, 0.0006632],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [20.87, 7.382, 0.9674, 0.1864, 1.802, 0.8999, 0.3166, 0.2959, 0.6901, 1.871, 1.753e-07, 0.0006331],
            # 2: [20.87, 6.984, 0.9483, 0.1938, 1.724, 0.8241, 0.3157, 0.2967, 0.6346, 1.901, 1.935e-07, 0.0006181]
        }
    },
    'BIV_S10': {
        "flat_asym_C4_q0p7": {
            1: [0.3141, 0.4021, 0.7549, 1, 2.391, 1.84, 0.02373, 0.000775]
        },
        'ULV_spheric_asym_C4': {
            1: [9.793, 6.719, 0.8183, 1.843, 0.3249, 0.2602, 0.6434, 1.882, 0.0002879, 0.0007089],
            # forced to have higher maximum on left side and larger radius:
            2: [17.48, 6.869, 1.703, 1.33, 0.1981, 0.4524, 0.814, 1.807, 0.0001049, 0.0008936],
            3: [9.291, 6, 1.251, 1.4, 0.223, 0.345, 0.6594, 1.873, 0.0003328, 0.0008027],
            4: [18.73, 6, 1.6, 1.6, 0.3829, 0.1361, 0.6638, 1.871, 0.0001063, 0.0008067],
            5: [18.58, 6, 2.128, 0.7, 0.2845, 0.3021, 0.649, 1.879, 0.0001061, 0.0007439],
            6: [18.58, 6, 1.796, 1, 0.3383, 0.2135, 0.654, 1.875, 0.0001071, 0.0007939],
            7: [9.53, 5.999, 1.2, 1.666, 0.3801, 0.3992, 1, 1.792, 0.0002574, 0.000781],
            8: [8.108, 6.967, 0.9565, 1.6, 0.2861, 0.3002, 0.6529, 1.876, 0.0003554, 0.0007652],
            9: [17.57, 6.745, 2.007, 1, 0.4141, 0.3563, 1, 1.793, 9.291e-05, 0.0007899],
            10: [15.33, 7.439, 1.941, 0.8001, 0.3151, 0.2657, 0.6552, 1.875, 0.0001399, 0.0007809],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [12.76, 0.497, 0.8789, 1.7, 0.2959, 0.2982, 0.6567, 1.882, 3.042e-07, 0.0007609],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [9.835, 6.916, 1.066, 0.02601, 0.8369, 1.826, 0.315, 0.2664, 0.6469, 1.842, 5.358e-07, 0.0006962],
            # 2: [9.777, 6.853, 1.014, 0.06389, 0.8536, 1.799, 0.3155, 0.2743, 0.6594, 1.871, 5.622e-07, 0.0007022],
        }
    },
    'BIV_S11': {
        'ULV_spheric_asym_C4': {
            1: [16.43, 2.896, 1.397, 0.9383, 0.3527, 0.3241, 0.6752, 1.902, 0.0001475, 0.0005264],
            2: [9.063, 4.568, 1.287, 1.031, 0.3328, 0.3483, 0.6798, 1.9, 0.000399, 0.000525],
            3: [15.61, 3.023, 1.223, 1.2, 0.3999, 0.2506, 0.6736, 1.901, 0.0001648, 0.0005061],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.12, 0.4463, 1.206, 1.1, 0.3388, 0.3431, 0.6817, 1.903, 6.626e-07, 0.0005266],
            2: [15.53, 0.1676, 1.41, 0.9437, 0.3444, 0.3156, 0.6658, 1.905, 3.283e-07, 0.0005247],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S12': {
        'ULV_spheric_asym_C4': {
            1: [10.08, 4.466, 1.148, 1.129, 0.3627, 0.318, 0.6704, 1.907, 0.0003359, 0.0005405],
            2: [16.11, 2.776, 1.326, 1, 0.3568, 0.2934, 0.6537, 1.91, 0.0001565, 0.0005422],
            3: [9, 5.223, 1.398, 0.9256, 0.3331, 0.3422, 0.6687, 1.907, 0.0003811, 0.0005462],
            4: [9, 4.534, 1.2, 1.2, 0.3408, 0.2801, 0.6417, 1.92, 0.0004127, 0.0005093],
            5: [16.09, 2.796, 1.578, 0.821, 0.2986, 0.3561, 0.6506, 1.912, 0.0001565, 0.0005276],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.31, 0.4503, 1.11, 1.196, 0.3572, 0.2997, 0.654, 1.918, 6.418e-07, 0.0005303],
            2: [15.59, 0.1623, 1.552, 0.8547, 0.2991, 0.3373, 0.6399, 1.916, 3.319e-07, 0.0005284],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S13': {
        'ULV_spheric_asym_C4': {
            1: [16.64, 2.81, 1.554, 0.8277, 0.3076, 0.3606, 0.6585, 1.91, 0.0001444, 0.0005327],
            2: [9.631, 4.61, 1.225, 1.043, 0.3425, 0.34, 0.6684, 1.904, 0.0003557, 0.0005084],
            3: [16.06, 3.002, 1.2, 1.2, 0.3961, 0.2463, 0.6558, 1.907, 0.0001547, 0.0004883],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.235, 0.4599, 1.081, 1.246, 0.3715, 0.2953, 0.6668, 1.913, 6.201e-07, 0.0004774]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S14': {
        'ULV_spheric_asym_C4': {
            1: [16.19, 3.432, 1.399, 0.9313, 0.3495, 0.3268, 0.6713, 1.898, 0.0001511, 0.0004995],
            2: [8.979, 5.075, 1.29, 1.028, 0.3267, 0.3464, 0.6698, 1.898, 0.0003915, 0.0004974],
            3: [15.31, 3.628, 1.226, 1.2, 0.3919, 0.2471, 0.6597, 1.901, 0.0001709, 0.0004789],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.208, 0.4844, 1.221, 1.1, 0.326, 0.3415, 0.6717, 1.9, 6.142e-07, 0.0004901]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S15': {
        'ULV_spheric_asym_C4': {
            1: [9.632, 5.068, 1.131, 1.167, 0.3644, 0.3211, 0.6787, 1.896, 0.0003442, 0.0004955],
            2: [15.27, 3.894, 1.4, 0.9442, 0.3508, 0.3141, 0.6667, 1.898, 0.0001673, 0.0005013],
            3: [14.39, 4.3, 1.236, 1.2, 0.4051, 0.2517, 0.6807, 1.892, 0.0001827, 0.0004827],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.622, 0.4824, 1.002, 1.335, 0.3895, 0.2894, 0.6745, 1.901, 5.583e-07, 0.0004697]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S16': {
        'ULV_spheric_asym_C4': {
            1: [15.58, 4.11, 1.443, 0.8991, 0.3415, 0.3208, 0.6544, 1.899, 0.0001751, 0.0005865],
            2: [7.388, 5.603, 1.246, 1.071, 0.3281, 0.3274, 0.6507, 1.9, 0.0005328, 0.0005785],
            3: [14.71, 2.44, 1.283, 1.2, 0.3703, 0.2209, 0.6312, 1.902, 0.000216, 0.00056],
            4: [12.4, 5.002, 1.265, 1.2, 0.3906, 0.2354, 0.6495, 1.9, 0.0002582, 0.0005628],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.5177, 1.14, 1.155, 0.3194, 0.3237, 0.6367, 1.909, 6.745e-07, 0.0005783],
            2: [13.73, 0.2859, 1.571, 0.8376, 0.3025, 0.3297, 0.6288, 1.91, 4.255e-07, 0.0005691],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S17': {
        'ULV_spheric_asym_C4': {
            1: [9, 5.564, 1.047, 1.315, 0.3517, 0.3133, 0.6728, 1.889, 0.0003872, 0.0005146],
            2: [15.17, 5.178, 1.4, 1.115, 0.3818, 0.2424, 0.666, 1.888, 0.0001728, 0.0005267],
            3: [7.033, 6, 1.2, 1.2, 0.3098, 0.331, 0.6635, 1.89, 0.000515, 0.0005193],
            4: [15.1, 5.2, 1.805, 0.757, 0.2944, 0.3547, 0.6605, 1.892, 0.0001725, 0.000497],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.4, 0.5001, 0.9628, 1.378, 0.3528, 0.3072, 0.6595, 1.895, 4.988e-07, 0.0005095],
            2: [15.98, 0.2966, 1.669, 0.8, 0.3105, 0.3347, 0.6502, 1.893, 2.959e-07, 0.0005156],
        },

        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S18': {
        'ULV_spheric_asym_C4': {
            1: [9, 5.944, 1.023, 1.357, 0.348, 0.3101, 0.6721, 1.888, 0.0004002, 0.0006703],
            2: [7.344, 6, 1.2, 1.21, 0.2993, 0.3373, 0.6648, 1.888, 0.0005254, 0.0006842],
            3: [14.83, 5.697, 1.4, 1.2, 0.3883, 0.2202, 0.6674, 1.886, 0.000189, 0.0006824],
            4: [15.02, 5.619, 1.723, 0.8, 0.3145, 0.3297, 0.6647, 1.889, 0.0001832, 0.0006723],
            5: [16, 5.481, 1.649, 1, 0.4119, 0.3609, 0.8864, 1.822, 0.0001409, 0.0006827],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.79, 0.5046, 0.9655, 1.41, 0.3527, 0.317, 0.6837, 1.882, 4.795e-07, 0.0006638],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S19': {
        'ULV_spheric_asym_C4': {
            1: [17.25, 5.844, 1.854, 0.8, 0.3075, 0.2945, 0.6527, 1.889, 0.000132, 0.0007921],
            2: [7.924, 6, 1.083, 1.4, 0.2874, 0.3139, 0.651, 1.888, 0.0004458, 0.000801],
            3: [16.73, 6, 1.4, 1.4, 0.4124, 0.1724, 0.6672, 1.877, 0.0001426, 0.0008409],
            4: [7.945, 6, 0.8812, 1.751, 0.3448, 0.247, 0.6371, 1.896, 0.0004459, 0.0007184],
            5: [8.382, 6, 1.2, 1.583, 0.4171, 0.3907, 1, 1.799, 0.0003331, 0.0007952],
            6: [11.74, 8, 1.4, 1.251, 0.4218, 0.2, 0.6879, 1.868, 0.0002201, 0.0008498],
            7: [15.64, 6.195, 1.668, 1, 0.3425, 0.2333, 0.6513, 1.887, 0.0001566, 0.0008133],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.77, 0.4911, 0.8732, 1.647, 0.3364, 0.2853, 0.6578, 1.891, 3.884e-07, 0.0007633],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S20': {
        'ULV_spheric_asym_C4': {
            1: [20.82, 7.417, 1.836, 0.8, 0.3039, 0.2889, 0.6452, 1.889, 9.574e-05, 0.0009371],
            2: [8.102, 6, 1.077, 1.4, 0.2818, 0.3162, 0.6469, 1.888, 0.0004597, 0.0009528],
            3: [17.21, 6, 1.4, 1.4, 0.4135, 0.169, 0.663, 1.877, 0.0001442, 0.0009972],
            4: [16.57, 6, 2.054, 0.7, 0.2769, 0.3228, 0.6404, 1.892, 0.0001504, 0.0009065],
            5: [16.56, 6, 1.67, 1, 0.3424, 0.2281, 0.6461, 1.888, 0.0001517, 0.0009644],
            6: [13.01, 8, 1.4, 1.227, 0.4235, 0.2, 0.6839, 1.868, 0.000202, 0.00101],
            7: [10.78, 8.66, 1.669, 1, 0.3377, 0.2281, 0.6432, 1.888, 0.0002486, 0.0009609],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.97, 0.5147, 0.8, 1.817, 0.3484, 0.2573, 0.6416, 1.9, 3.846e-07, 0.0008729],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S21': {
        'ULV_spheric_asym_C4': {
            1: [16.16, 2.794, 1.386, 0.9551, 0.3473, 0.308, 0.6549, 1.911, 0.0001761, 0.0005239],
            2: [8.534, 4.574, 1.2, 1.2, 0.3473, 0.2816, 0.6461, 1.919, 0.0005078, 0.0004865],
            # 1: [9.439, 4.549, 1.285, 1.029, 0.327, 0.3481, 0.6689, 1.909, 0.0004254, 0.0005276]
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4542, 1.206, 1.097, 0.3365, 0.34, 0.6737, 1.907, 7.632e-07, 0.0005132],
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [9.371, 4.747, 0.9896, 0.1953, 1.347, 1.097, 0.317, 0.338, 0.6874, 1.889, 8.572e-07, 0.0005051],
            # 2: [9.618, 4.747, 1.136, 0.1462, 1.219, 1.129, 0.3172, 0.3381, 0.6633, 1.809, 7.41e-07, 0.0004558]
        }
    },
    'BIV_S22': {
        'ULV_spheric_asym_C4': {
            1: [16.12, 2.826, 1.511, 0.8386, 0.3237, 0.3622, 0.6723, 1.903, 0.0001828, 0.0005951],
            2: [8.929, 4.633, 1.215, 1.062, 0.3599, 0.3341, 0.6817, 1.9, 0.0004821, 0.000603],
            3: [15.75, 2.603, 1.21, 1.2, 0.3985, 0.2456, 0.6624, 1.904, 0.0001968, 0.0005792],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4482, 1.162, 1.094, 0.3596, 0.3446, 0.6885, 1.899, 8.037e-07, 0.0006116]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.05, 3.025, 0.8926, 0.2745, 1.533, 0.9026, 0.3138, 0.3534, 0.6768, 1.909, 4.314e-07, 0.0005479],
            # 1: [16.32, 3.025, 1.11, 0.2161, 1.47, 0.8752, 0.3151, 0.3522, 0.6483, 1.816, 3.425e-07, 0.0005312]
        }
    },
    'BIV_S23': {
        'ULV_spheric_asym_C4': {
            1: [16.14, 2.666, 1.361, 0.9862, 0.3525, 0.2982, 0.6554, 1.907, 0.0001876, 0.0005509],
            2: [8.747, 4.499, 1.335, 1.004, 0.321, 0.3443, 0.6653, 1.905, 0.0005152, 0.0005511],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4584, 1.202, 1.072, 0.3313, 0.3374, 0.6563, 1.912, 8.212e-07, 0.0005569]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [15.97, 2.866, 0.9517, 0.2707, 1.38, 1.081, 0.3424, 0.2889, 0.6609, 1.885, 4.147e-07, 0.0005221]
        }
    },
    'BIV_S24': {
        'ULV_spheric_asym_C4': {
            1: [16.14, 2.714, 1.363, 0.9531, 0.3528, 0.3093, 0.6559, 1.904, 0.0002031, 0.0005877],
            2: [8.846, 4.454, 1.304, 1.002, 0.3258, 0.3467, 0.6636, 1.902, 0.0005524, 0.0005872],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4656, 1.195, 1.087, 0.3421, 0.3427, 0.6755, 1.901, 8.569e-07, 0.0005814]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [15.95, 2.914, 0.9734, 0.1933, 1.399, 1.021, 0.3429, 0.2993, 0.6615, 1.898, 4.301e-07, 0.000578]
        }
    },
    'BIV_S25': {
        'ULV_spheric_asym_C4': {
            1: [15.93, 3.017, 1.378, 0.9368, 0.3497, 0.3141, 0.6549, 1.903, 0.0002164, 0.0006444],
            2: [8.615, 4.698, 1.29, 1.013, 0.3273, 0.3435, 0.6605, 1.902, 0.0005885, 0.0006419],
            3: [15.48, 2.61, 1.233, 1.2, 0.3837, 0.2357, 0.6422, 1.905, 0.000237, 0.0006136],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4827, 1.255, 1.056, 0.3175, 0.3542, 0.6682, 1.899, 8.729e-07, 0.000637]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.73, 2.817, 0.8714, 0.06079, 1.478, 1.037, 0.367, 0.324, 0.7541, 1.954, 4.747e-07, 0.0005007],
            # 2: [15.77, 3.213, 0.9435, 0.2109, 1.44, 1.011, 0.3397, 0.3078, 0.6749, 1.906, 4.64e-07, 0.0006092],
            # 3: [16.07, 3.216, 1.146, 0.2052, 1.338, 0.949, 0.34, 0.3127, 0.6291, 1.803, 3.961e-07, 0.0006117]
        }
    },
    'BIV_S26': {
        'ULV_spheric_asym_C4': {
            1: [15.47, 3.765, 1.475, 0.8843, 0.3333, 0.3322, 0.6582, 1.901, 0.0002034, 0.0006327],
            2: [14.92, 2.429, 1.4, 0.973, 0.3399, 0.2801, 0.6362, 1.904, 0.000236, 0.0006238],
            3: [14.52, 3.93, 1.256, 1.2, 0.3883, 0.24, 0.6517, 1.902, 0.0002317, 0.0006103],
            4: [7.967, 5.188, 1.236, 1.086, 0.3372, 0.3246, 0.6574, 1.901, 0.0005726, 0.0006272],
            # 1: [9, 5.06, 1.221, 1.086, 0.336, 0.3382, 0.6656, 1.9, 0.0004797, 0.0006352]
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.122, 0.5009, 1.155, 1.15, 0.3256, 0.3244, 0.6445, 1.909, 7.669e-07, 0.0006205]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [8.807, 5.248, 0.9859, 0.09175, 1.249, 1.137, 0.326, 0.3282, 0.6726, 1.899, 1.026e-06, 0.0005477]
        }
    },
    'BIV_S27': {
        'ULV_spheric_asym_C4': {
            1: [15.3, 4.563, 1.575, 0.843, 0.3269, 0.3381, 0.6638, 1.894, 0.0001994, 0.0006189],
            2: [6.425, 6, 1.22, 1.131, 0.3304, 0.3218, 0.6564, 1.895, 0.0006665, 0.0006155],
            3: [14.19, 2.558, 1.4, 1.048, 0.3516, 0.2467, 0.6361, 1.897, 0.0002609, 0.0006129],
            4: [14.09, 4.798, 1.307, 1.2, 0.3941, 0.2341, 0.6611, 1.893, 0.0002331, 0.0006079],
            # 2: [9, 5.376, 1.139, 1.186, 0.3437, 0.3312, 0.6719, 1.892, 0.0004574, 0.0006214]
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.5279, 1.206, 1.131, 0.3044, 0.3573, 0.6666, 1.891, 7.265e-07, 0.0006279]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [9.021, 5.541, 0.8988, 0.2963, 1.134, 1.171, 0.3341, 0.3257, 0.6508, 1.889, 1.111e-06, 0.0006048],
            # 2: [9.146, 5.575, 1.111, 0.1403, 1.118, 1.286, 0.3338, 0.3248, 0.6826, 1.803, 8.026e-07, 0.000538],
            # 3: [9.147, 5.548, 0.9825, 0.2466, 1.077, 1.159, 0.3345, 0.3264, 0.6303, 1.875, 1.002e-06, 0.0004942]
        }
    },
    'BIV_S28': {
        'ULV_spheric_asym_C4': {
            1: [15.42, 4.433, 1.558, 0.8492, 0.3361, 0.3338, 0.6703, 1.888, 0.0002257, 0.0008178],
            2: [6.348, 5.821, 1.223, 1.124, 0.3282, 0.3314, 0.664, 1.889, 0.000792, 0.0008129],
            3: [14.33, 4.616, 1.321, 1.2, 0.3933, 0.2304, 0.6613, 1.889, 0.0002619, 0.0007995],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.039, 0.5272, 1.149, 1.184, 0.3163, 0.3467, 0.6689, 1.888, 8.233e-07, 0.0008198]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [15.36, 4.631, 0.9491, 0.2631, 1.609, 0.9179, 0.3263, 0.324, 0.6809, 1.869, 4.88e-07, 0.0007793],
            # 2: [15.57, 4.632, 1.132, 0.1746, 1.519, 0.8601, 0.3305, 0.3242, 0.6442, 1.795, 4.163e-07, 0.0007935]
        }
    },
    'BIV_S29': {
        'ULV_spheric_asym_C4': {
            1: [16.83, 5.413, 1.714, 0.8, 0.3243, 0.3126, 0.6564, 1.888, 0.0001515, 0.0007714],
            2: [7.42, 6, 1.2, 1.235, 0.2776, 0.3418, 0.6525, 1.888, 0.0005247, 0.0007761],
            3: [15.84, 5.652, 1.4, 1.261, 0.4016, 0.2, 0.6656, 1.881, 0.0001696, 0.0007813],
            4: [9.611, 8, 1.4, 1.362, 0.3913, 0.183, 0.6542, 1.886, 0.000312, 0.0007639],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.88, 0.5032, 1.069, 1.328, 0.2891, 0.3372, 0.652, 1.89, 4.895e-07, 0.0007787]
        },
        'ULV_ellipsoid_asym_C4': {
            # 1: [16.98, 5.612, 1.137, 0.1337, 1.731, 0.8443, 0.3161, 0.3028, 0.6542, 1.79, 2.681e-07, 0.0007421],
            # 2: [16.86, 5.613, 0.8785, 0.292, 1.702, 0.8753, 0.329, 0.3026, 0.6707, 1.89, 3.57e-07, 0.0007345]
        }
    },
    'BIV_S30': {
        'ULV_spheric_asym_C4': {
            1: [16.8, 6.039, 1.861, 0.8, 0.3081, 0.2829, 0.6389, 1.891, 0.0001453, 0.000664],
            2: [7.814, 6, 1.2, 1.353, 0.249, 0.3254, 0.6359, 1.89, 0.0004811, 0.0006757],
            3: [16.11, 6, 1.6, 1.136, 0.361, 0.2, 0.6443, 1.886, 0.0001583, 0.0006923],
            4: [7.861, 6, 0.9255, 1.628, 0.3182, 0.2682, 0.6277, 1.896, 0.0004767, 0.000617],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.63, 0.5217, 0.9583, 1.533, 0.2879, 0.3044, 0.6339, 1.896, 4.041e-07, 0.0006682]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.99, 6.239, 1.168, 0.01873, 1.82, 0.8121, 0.2981, 0.2729, 0.6101, 1.791, 2.583e-07, 0.000682],
            # 2: [16.65, 6.074, 0.8248, 0.3252, 1.849, 0.8581, 0.3066, 0.2729, 0.6399, 1.894, 3.921e-07, 0.0006033]
        }
    },
    'BIV_S31': {
        'ULV_spheric_asym_C4': {
            1: [16, 2.684, 1.344, 1.008, 0.3486, 0.2828, 0.6361, 1.919, 0.0001774, 0.0004727],
            2: [8.666, 4.5, 1.353, 0.9918, 0.3078, 0.3385, 0.6437, 1.919, 0.0004861, 0.000474],
            3: [15.71, 2.422, 1.255, 1.2, 0.3693, 0.2317, 0.6294, 1.92, 0.000188, 0.0004538],
            4: [9, 4.394, 1.2, 1.2, 0.3349, 0.2783, 0.6292, 1.928, 0.0004653, 0.0004417],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4461, 1.265, 1.071, 0.3149, 0.337, 0.654, 1.918, 7.586e-07, 0.0004634],
            2: [15.51, 0.1408, 1.346, 1.059, 0.3464, 0.265, 0.6336, 1.92, 3.779e-07, 0.0004532],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S32': {
        'ULV_spheric_asym_C4': {
            1: [16.32, 2.602, 1.456, 0.8964, 0.3329, 0.3359, 0.666, 1.905, 0.0001653, 0.0004612],
            2: [9.279, 4.324, 1.284, 1.029, 0.3345, 0.3424, 0.6736, 1.903, 0.0004306, 0.0004668],
            3: [15.99, 2.48, 1.232, 1.2, 0.3879, 0.2443, 0.6567, 1.906, 0.0001764, 0.0004448],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4518, 1.205, 1.1, 0.3392, 0.3288, 0.6655, 1.911, 7.481e-07, 0.0004602],
            2: [15.58, 0.1442, 1.448, 0.9174, 0.3252, 0.3176, 0.6481, 1.912, 3.683e-07, 0.0004556],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S33': {
        'ULV_spheric_asym_C4': {
            1: [16.13, 2.813, 1.464, 0.8676, 0.3329, 0.3475, 0.6686, 1.901, 0.0001671, 0.000532],
            2: [8.804, 4.69, 1.226, 1.057, 0.3559, 0.3338, 0.6788, 1.899, 0.0004479, 0.0005336],
            3: [15.77, 2.541, 1.218, 1.2, 0.3945, 0.2433, 0.6583, 1.903, 0.0001801, 0.0005119],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4823, 1.164, 1.11, 0.3637, 0.3467, 0.7018, 1.892, 6.782e-07, 0.0005333]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S34': {
        'ULV_spheric_asym_C4': {
            1: [15.9, 2.9, 1.38, 0.9403, 0.3547, 0.3214, 0.6704, 1.898, 0.0001752, 0.0005487],
            2: [8.7, 4.631, 1.301, 1.01, 0.3357, 0.3495, 0.6783, 1.897, 0.0004687, 0.0005474],
            3: [15.52, 2.578, 1.228, 1.2, 0.3915, 0.2428, 0.6575, 1.901, 0.0001895, 0.0005232],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4682, 1.176, 1.108, 0.3429, 0.3348, 0.6687, 1.905, 7.327e-07, 0.0005434]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S35': {
        'ULV_spheric_asym_C4': {
            1: [15.73, 3.565, 1.418, 0.9045, 0.351, 0.3396, 0.6794, 1.892, 0.0001719, 0.0005669],
            2: [8.638, 5.082, 1.234, 1.065, 0.3494, 0.3402, 0.6814, 1.892, 0.0004452, 0.0005601],
            3: [14.86, 3.747, 1.223, 1.2, 0.4023, 0.2502, 0.6701, 1.895, 0.000194, 0.00054],
            4: [15.14, 2.485, 1.345, 1, 0.3582, 0.2857, 0.6545, 1.896, 0.0001981, 0.0005542],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.298, 0.4761, 1.186, 1.085, 0.3362, 0.3511, 0.6738, 1.895, 6.672e-07, 0.0005693],
            2: [14.26, 0.2525, 1.389, 0.9224, 0.3518, 0.3289, 0.6688, 1.896, 3.96e-07, 0.0005697],
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S36': {
        'ULV_spheric_asym_C4': {
            1: [15.61, 3.762, 1.532, 0.8311, 0.3222, 0.3504, 0.6566, 1.899, 0.0001824, 0.0005471],
            2: [7.36, 5.25, 1.2, 1.2, 0.339, 0.283, 0.6362, 1.909, 0.0005865, 0.0005125],
            3: [14.73, 3.892, 1.244, 1.2, 0.3935, 0.2376, 0.65, 1.9, 0.0002065, 0.0005326],
            4: [14.96, 2.436, 1.4, 0.9486, 0.3404, 0.2836, 0.6335, 1.901, 0.000215, 0.0005463],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.4892, 1.137, 1.139, 0.3252, 0.3263, 0.6376, 1.909, 7.4e-07, 0.0005551]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S37': {
        'ULV_spheric_asym_C4': {
            1: [15.67, 4.279, 1.525, 0.8607, 0.3404, 0.3371, 0.6746, 1.887, 0.000172, 0.0005896],
            2: [6.929, 5.7, 1.257, 1.081, 0.3238, 0.344, 0.669, 1.888, 0.0005703, 0.000589],
            3: [14.26, 2.668, 1.315, 1.2, 0.3758, 0.2187, 0.6424, 1.892, 0.0002309, 0.0005691],
            4: [16, 4.073, 1.29, 1.2, 0.3981, 0.2369, 0.6649, 1.889, 0.0001698, 0.000575],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9, 0.5338, 1.195, 1.147, 0.3191, 0.3652, 0.6954, 1.877, 6.278e-07, 0.0005889]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S38': {
        'ULV_spheric_asym_C4': {
            1: [15.46, 4.586, 1.643, 0.8074, 0.3152, 0.3487, 0.6621, 1.893, 0.0001983, 0.0006792],
            2: [14.34, 4.779, 1.4, 1.021, 0.3681, 0.2705, 0.6561, 1.893, 0.0002299, 0.0006953],
            3: [13.99, 2.676, 1.4, 1.064, 0.3536, 0.2398, 0.6343, 1.895, 0.0002722, 0.0006833],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.045, 0.5402, 1.063, 1.27, 0.341, 0.3236, 0.6693, 1.89, 7.049e-07, 0.000679]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S39': {
        'ULV_spheric_asym_C4': {
            1: [15.52, 5.349, 1.69, 0.8, 0.3266, 0.3288, 0.6658, 1.884, 0.0002085, 0.0008922],
            2: [6.462, 6, 1.029, 1.358, 0.3453, 0.2985, 0.6539, 1.887, 0.0007319, 0.0008657],
            3: [14.1, 5.587, 1.4, 1.213, 0.3941, 0.2133, 0.6615, 1.883, 0.0002499, 0.0008945],
            4: [9, 7.632, 1.706, 0.8, 0.3156, 0.3285, 0.6584, 1.885, 0.0004107, 0.000883],
            5: [9, 7.722, 1.4, 1.155, 0.3852, 0.2284, 0.6613, 1.883, 0.000411, 0.0008946],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.818, 0.5442, 0.9547, 1.445, 0.3518, 0.3113, 0.6786, 1.881, 6.387e-07, 0.0008634]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BIV_S40': {
        'ULV_spheric_asym_C4': {
            1: [9, 6.243, 0.9272, 1.535, 0.3476, 0.2945, 0.6659, 1.881, 0.0004576, 0.0008808],
            2: [15.75, 6, 1.4, 1.4, 0.4156, 0.1771, 0.6722, 1.872, 0.0002006, 0.0009463],
            3: [7.526, 6, 1.2, 1.296, 0.272, 0.3409, 0.6622, 1.879, 0.0006014, 0.0009257],
            4: [15.32, 6, 1.6, 1, 0.3629, 0.2447, 0.6664, 1.877, 0.0002066, 0.0009404],
            5: [15.26, 6, 2.018, 0.7, 0.2851, 0.3441, 0.6567, 1.883, 0.0002063, 0.0008659],
            6: [10.16, 8, 1.4, 1.278, 0.4196, 0.2, 0.6877, 1.866, 0.0003378, 0.0009526],
            7: [10.15, 8, 2.042, 0.7, 0.2776, 0.3435, 0.6524, 1.884, 0.0003361, 0.0008574],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.7, 0.5328, 0.8911, 1.563, 0.3464, 0.305, 0.6757, 1.877, 5.457e-07, 0.0008902]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [],
        }
    },
    'BVI_S1': {
        "ULV_sph_asym_prot_layer": {
            1: [9.003, 7.646, 1.359, 1.235, 0.3743, 0.4144, 0.962, 1.718, 3.001, 0.7564, 0.1721, 4.543e-07, 0.0005787], # chi2 = 99
            2: [9.012, 7.796, 0.9664, 1.372, 0.4288, 0.3136, 0.7846, 1.752, 3.007, 0.7115, 0.1663, 4.839e-07, 0.0005676],
        },
        "ULV_lognormal_sph_asym_prot_layer": {
            1: [13.67, 0.48, 1.046, 1.318, 0.3509, 0.2744, 0.7, 1.768, 2.848, 0.7, 0.2214, 2.377e-07, 0.0005671], # chi2 = 64
            2: [13.63, 0.4824, 1.337, 1.533, 0.4605, 0.4295, 1.2, 1.711, 3.209, 0.5001, 0.2417, 2.105e-07, 0.0005758], # chi2 = 65
            3: [13.72, 0.4756, 1.05, 1.125, 0.3593, 0.2249, 0.7693, 1.765, 2, 1.122, 0.2798, 2.648e-07, 0.0005753], # chi2 = 66
            4: [15, 0.4802, 1.201, 1.106, 0.303, 0.302, 0.7037, 1.763, 2.68, 0.7901, 0.2379, 1.985e-07, 0.0005794], # chi2 = 67
            5: [15, 0.4785, 1.192, 0.9819, 0.3184, 0.2558, 0.7699, 1.757, 2.093, 1.03, 0.3072, 2.107e-07, 0.0005769], # chi2 = 68
            6: [15, 0.4755, 1.022, 1.277, 0.3942, 0.2227, 0.8286, 1.75, 2, 1.101, 0.29, 2.121e-07, 0.0005576], # chi2 = 69
        }
    },
    'BVI_S2': {
        'ULV_spheric_asym_C4': {
            # 1: [19.06, 7.757, 1.006, 2, 0.778, 0.1797, 0.9239, 1.741, 9.037e-05, 0.0004566],
            # see C:\Users\riko05\Documents\8. SoSe 26\Code\BM29_data_and_analysis_2026\analysis_BM29_2026\Results_and_Figures\Sep11\sph_asym_ambiguity_BVI\BVI_S2
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.565, 8.549, 1.147, 1.154, 0.319, 0.3441, 0.7101, 1.769, 3.001, 0.7, 0.1724, 4.327e-07, 0.0005452], # chi2 = 92
            2: [9.032, 8.189, 0.9198, 1.302, 0.4335, 0.3475, 0.7814, 1.754, 3.007, 0.701, 0.1501, 4.625e-07, 0.0005643],
        },
        "ULV_lognormal_sph_asym_prot_layer": {
            1: [14.18, 0.4679, 1.03, 1.454, 0.3651, 0.2627, 0.734, 1.756, 2.808, 0.7, 0.2308, 2.164e-07, 0.0005088], # chi2 = 70.6
            2: [15, 0.4646, 1.007, 1.59, 0.3601, 0.1996, 0.7153, 1.762, 2.519, 0.8371, 0.2598, 2.004e-07, 0.0005012], # chi2 = 71.4
            3: [15, 0.4635, 1.547, 0.8286, 0.2478, 0.3457, 0.7793, 1.749, 2.203, 1.032, 0.2796, 2.086e-07, 0.0005166], # chi2 = 72.4
            4: [15, 0.4628, 1.063, 1.349, 0.3893, 0.2182, 0.8605, 1.741, 2, 1.059, 0.3134, 2.085e-07, 0.0005069], # chi2 = 72.9
        }
    },
    'BVI_S3': {
        'ULV_spheric_asym_C4': {
            # 1: [15.93, 7.882, 1.046, 2, 0.7708, 0.1788, 0.9389, 1.738, 0.0001255, 0.0004763],
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.003, 7.6, 1.12, 1.25, 0.3716, 0.375, 0.8249, 1.745, 3.005, 0.7823, 0.1483, 4.973e-07, 0.0005717],
            # 2: [13.37, 8.014, 0.8269, 1.956, 0.4497, 0.2329, 0.7611, 1.772, 3.026, 0.7457, 0.1465, 2.858e-07, 0.0004241],
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": {
            1: [14.59, 0.4488, 0.9456, 1.572, 0.3896, 0.2445, 0.7398, 1.75, 2.719, 0.7, 0.2429, 2.194e-07, 0.0005249], # chi2 = 64.5
            2: [15, 0.446, 1.052, 1.32, 0.3628, 0.3117, 0.7719, 1.743, 2.758, 0.72, 0.2253, 2.06e-07, 0.0005589], # chi2 = 65.4
            3: [15, 0.4461, 1.218, 0.9361, 0.2892, 0.2421, 0.7508, 1.758, 2.008, 0.9997, 0.342, 2.396e-07, 0.0005677], # chi2 = 65.8
            4: [15, 0.4456, 1.079, 1.19, 0.3324, 0.2002, 0.7645, 1.758, 2, 0.9988, 0.3395, 2.361e-07, 0.0005598], # chi2 = 66
            5: [14.39, 0.4529, 1.495, 0.9848, 0.2852, 0.2029, 0.8861, 1.716, 2, 0.8782, 0.5017, 2.265e-07, 0.0005273], # chi2 = 73.7
            6: [13.88, 0.4484, 1.267, 1.278, 0.4329, 0.274, 1.128, 1.702, 2.003, 1.03, 0.4004, 2.441e-07, 0.0005307], # chi2 = 73.8
        }
    },
    'BVI_S4': {
        'ULV_spheric_asym_C4': {
            # 1: [8.193, 7.98, 3.438, 0.9101, 0.13, 0.741, 0.956, 1.747, 0.00029, 0.0004536],
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.031, 7.667, 1.381, 1.476, 0.4262, 0.4236, 1.172, 1.704, 3.001, 0.712, 0.1976, 4.649e-07, 0.000613],
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": {
            1: [14.54, 0.4424, 0.988, 1.523, 0.3566, 0.2561, 0.7222, 1.763, 2.751, 0.7, 0.2311, 2.273e-07, 0.0005712],
            2: [15, 0.4406, 1.554, 1.348, 0.3511, 0.4685, 1.122, 1.714, 3.02, 0.7005, 0.1933, 1.928e-07, 0.0006002],
            3: [15, 0.442, 1.448, 1.227, 0.3895, 0.3647, 1.159, 1.706, 2.232, 0.941, 0.3583, 1.941e-07, 0.0005933],
            4: [15, 0.4383, 1.191, 1.021, 0.2832, 0.2419, 0.7367, 1.773, 2, 1.049, 0.3061, 2.494e-07, 0.0006043],
            5: [15, 0.4377, 1.061, 1.306, 0.3989, 0.2731, 0.911, 1.735, 2.004, 1.062, 0.2924, 2.214e-07, 0.0005809],
        }
    },
    'BVI_S5': {
        'ULV_spheric_asym_C4': {
            # 1: [8.065, 8.069, 3.685, 1.038, 0.13, 0.7377, 1.081, 1.729, 0.0002737, 0.00052],
            # Nothing acceptable
        },
        "ULV_spheric_asym_C4_lognormal": {
            # 1: [15.67, 0.449, 2.155, 1.263, 0.1784, 0.6, 0.9563, 1.773, 1.663e-07, 0.0006841],
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.792, 7.835, 1.057, 1.753, 0.4637, 0.3844, 1.083, 1.722, 3.004, 0.7001, 0.1453, 3.76e-07, 0.0006854], # chi2 = 140
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # all roughly chi2 = 90
            1: [15.17, 0.4396, 0.8281, 2.229, 0.4447, 0.2046, 0.8436, 1.733, 2.431, 0.7, 0.3022, 1.755e-07, 0.000584],
            2: [15.17, 0.4396, 0.8453, 2.108, 0.4428, 0.2209, 0.856, 1.73, 2.442, 0.7, 0.298, 1.748e-07, 0.0005976],
            3: [15.01, 0.4463, 2.199, 1.228, 0.2474, 0.573, 1.2, 1.701, 3.004, 0.7, 0.1926, 1.911e-07, 0.0006159],
            4: [14.84, 0.4422, 0.8456, 1.579, 0.3696, 0.3164, 0.7267, 1.774, 3.056, 0.5003, 0.1926, 2.059e-07, 0.0006867],
            5: [15.1, 0.4451, 1.571, 1.35, 0.3505, 0.4308, 1.2, 1.705, 2.482, 0.8028, 0.3261, 1.693e-07, 0.000698],
        }
    },
    'BVI_S6': {
        'ULV_spheric_asym_C4': {
            # 1: [8.14, 8.019, 3.852, 1.095, 0.13, 0.7558, 1.167, 1.718, 0.000249, 0.000524],
            # 2: [8, 8.023, 1.582, 1.476, 0.2409, 0.516, 0.9576, 1.776, 0.0001878, 0.0007345],
            # Nothing acceptable
        },
        "ULV_spheric_asym_C4_lognormal": {
            # 1: [15.49, 0.4257, 2.323, 1.422, 0.1604, 0.5679, 0.9862, 1.771, 1.47e-07, 0.0007015],
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.88, 7.517, 0.9635, 1.446, 0.2936, 0.3579, 0.715, 1.785, 3, 0.7002, 0.1223, 3.846e-07, 0.0007303], # chi2 = 121
            # 2: [10.24, 7.663, 1.009, 1.624, 0.347, 0.3853, 0.8741, 1.751, 3, 0.7001, 0.1131, 3.218e-07, 0.000712],
            # 3: [10.1, 7.675, 0.8122, 1.662, 0.3401, 0.3045, 0.7004, 1.789, 3.002, 0.7001, 0.1275, 3.689e-07, 0.0006861],
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # chi2 between 76 and 79
            1: [15.16, 0.4215, 0.8375, 2.379, 0.4398, 0.2518, 0.9105, 1.729, 2.662, 0.5, 0.2912, 1.465e-07, 0.0006014],
            2: [15.14, 0.4217, 0.9159, 2.108, 0.4421, 0.2472, 0.9841, 1.716, 2.261, 0.7, 0.3723, 1.466e-07, 0.0006266],
            3: [15.12, 0.4217, 0.8729, 1.912, 0.3973, 0.2696, 0.8633, 1.736, 2.407, 0.7, 0.284, 1.541e-07, 0.0006563],
            4: [15.02, 0.4221, 0.9089, 1.607, 0.3252, 0.3193, 0.7486, 1.768, 2.743, 0.7007, 0.1835, 1.727e-07, 0.0007048],
            # 5: [15.14, 0.4253, 1.75, 1.461, 0.2942, 0.4994, 1.2, 1.721, 2.831, 0.7518, 0.1983, 1.558e-07, 0.0007014],
        }
    },
    'BVI_S7': { # 15 mM DOPC/GM1 + 100 uM aSynWT
        'ULV_spheric_asym_C4': {
            # 1: [15.01, 8.157, 2.336, 0.8852, 0.431, 0.3189, 0.9825, 1.769, 7.665e-05, 0.0009907],
            # Nothing acceptable
        },
        "ULV_spheric_asym_C4_lognormal": {
            # FOUND NOTHING YET
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.72, 7.706, 1.057, 1.921, 0.3509, 0.4045, 1.021, 1.748, 3, 0.7, 0.07649, 3.142e-07, 0.0009686],
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # chi2 between 101 and 104
            1: [15, 0.4238, 0.8554, 2.285, 0.4081, 0.2408, 0.9649, 1.73, 2.142, 0.7001, 0.3779, 1.472e-07, 0.0008601],
            2: [14.94, 0.4242, 0.8507, 2.235, 0.3991, 0.2535, 0.9419, 1.732, 2.212, 0.7, 0.3346, 1.492e-07, 0.0008687],
            3: [14.93, 0.4245, 0.8238, 2.114, 0.3304, 0.2927, 0.7953, 1.766, 2.807, 0.5, 0.1917, 1.605e-07, 0.0009022],
            4: [14.91, 0.4246, 0.8453, 2.005, 0.3325, 0.2797, 0.8122, 1.757, 2.427, 0.7, 0.2316, 1.601e-07, 0.0009076],
            5: [15.01, 0.4255, 1.07, 1.882, 0.4345, 0.3164, 1.2, 1.729, 2.023, 0.9127, 0.3616, 1.583e-07, 0.0009234],
        }
    },
    'BVI_S8': {
        'ULV_spheric_asym_C4': {
            # 1: [8.172, 8.215, 3.302, 0.8599, 0.13, 0.7582, 0.9167, 1.75, 0.0002749, 0.0004437],
            # Nothing acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.055, 7.386, 1.246, 1.025, 0.2854, 0.3963, 0.7012, 1.771, 3, 0.7004, 0.1573, 5.339e-07, 0.0006061], # chi2 = 116
            # Nothing acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # chi2 between 69 and 71
            1: [14.79, 0.4545, 0.8745, 1.677, 0.4172, 0.2675, 0.7438, 1.761, 3.04, 0.5, 0.2289, 2.035e-07, 0.0005118],
            2: [14.77, 0.4542, 0.9223, 1.711, 0.4124, 0.2378, 0.7685, 1.746, 2.732, 0.7, 0.2403, 1.981e-07, 0.0005126],
            3: [14.73, 0.4558, 1.499, 0.8, 0.242, 0.2947, 0.767, 1.752, 2.053, 0.9792, 0.3508, 2.252e-07, 0.0005563],
            4: [15, 0.4522, 1.139, 1.068, 0.3073, 0.2194, 0.7504, 1.764, 2, 1.022, 0.3304, 2.277e-07, 0.0005713],
        }
    },
    'BVI_S9': {
        'ULV_spheric_asym_C4': {
            # 1: [8, 8.302, 3.448, 0.9239, 0.13, 0.7432, 0.97, 1.743, 0.0003002, 0.0005358],
            # Not acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.167, 8.128, 0.9657, 1.378, 0.3673, 0.313, 0.7277, 1.769, 3.002, 0.7008, 0.161, 5.146e-07, 0.0006912], # chi2 = 108
            # 2: [9.271, 8.217, 1.371, 1.579, 0.4337, 0.4056, 1.2, 1.708, 3.007, 0.7002, 0.2104, 4.375e-07, 0.0006844],
            # Not acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # chi2 between 69 and 71
            1: [14.74, 0.4511, 0.933, 1.587, 0.3579, 0.2728, 0.7, 1.777, 3.061, 0.5, 0.2246, 2.356e-07, 0.0006483],
            2: [14.72, 0.4511, 0.9999, 1.563, 0.3376, 0.2417, 0.7005, 1.768, 2.754, 0.7, 0.2359, 2.338e-07, 0.0006561],
            3: [15, 0.4492, 1.213, 1.138, 0.2817, 0.2124, 0.7501, 1.767, 2.017, 0.9919, 0.341, 2.481e-07, 0.0006848],
            4: [14.5, 0.4523, 1.207, 1.388, 0.2914, 0.1626, 0.7668, 1.762, 2, 0.9591, 0.3785, 2.587e-07, 0.0006719],
        }
    },
    'BVI_S10': {
        'ULV_spheric_asym_C4': {
            # 1: [8.258, 7.811, 3.512, 0.9126, 0.13, 0.7579, 0.984, 1.738, 0.0003006, 0.0004869],
            # Not acceptable
        },
        "ULV_sph_asym_prot_layer": {
            1: [9.678, 8.1, 1.42, 1.203, 0.291, 0.4421, 0.8684, 1.746, 3, 0.7, 0.1454, 4.224e-07, 0.0006595], # chi2 = 112
            # 2: [9.212, 7.667, 0.8388, 1.613, 0.4466, 0.2837, 0.7667, 1.762, 3.005, 0.7001, 0.158, 4.959e-07, 0.0006084],
            # Not acceptable
        },
        "ULV_lognormal_sph_asym_prot_layer": { # chi2 between 77 and 82
            1: [14.59, 0.4504, 0.8603, 1.942, 0.4299, 0.2222, 0.7745, 1.752, 2.74, 0.7, 0.2279, 2.138e-07, 0.0005565],
            2: [15, 0.4464, 0.9681, 1.549, 0.4335, 0.2401, 0.9186, 1.736, 2, 1.065, 0.2886, 2.179e-07, 0.0005977],
            3: [15, 0.4465, 1.314, 0.939, 0.2896, 0.3616, 0.8444, 1.75, 2, 1.144, 0.2611, 2.316e-07, 0.0006513],
        }
    },
    'BVI_S11': {
        'ULV_spheric_asym_C4': { # chi2 80 to 87
            1: [18.97, 9.673, 2.277, 0.9919, 0.2508, 0.486, 0.789, 1.801, 0.000109, 0.0005184],
            2: [18.81, 9.686, 1.156, 1.956, 0.486, 0.2506, 0.7893, 1.801, 0.0001105, 0.0005189],
            3: [19.88, 8.5, 1.4, 1.4, 0.4005, 0.3435, 0.7784, 1.807, 0.0001053, 0.0006622],
            4: [19.39, 8.391, 1.874, 1.2, 0.335, 0.4504, 0.8967, 1.783, 0.0001042, 0.0006221],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [10.07, 0.7351, 1.207, 1.808, 0.4677, 0.2614, 0.7746, 1.807, 3.827e-07, 0.0005554], # chi2 = 74
        },
        'ULV_sph_asym_prot_layer': { # all quite good chi2 between 69 and 85
            1: [15.7, 10, 1.129, 1.892, 0.4573, 0.2494, 0.7428, 1.815, 4.999, 1.192, 0.01155, 3.007e-07, 0.0005299], # PREMIUM chi2 = 69
            2: [18.44, 10, 1.534, 1.76, 0.4819, 0.3651, 1.11, 1.768, 4.915, 1.167, 0.008602, 2.035e-07, 0.000613], # also very good
            3: [11.37, 9.996, 1.418, 1.862, 0.5131, 0.3202, 1.055, 1.764, 5, 1.161, 0.01219, 4.083e-07, 0.0005523],
            4: [12.18, 10, 1.16, 1.894, 0.4553, 0.2395, 0.7344, 1.818, 4.959, 1.198, 0.01253, 4.224e-07, 0.0005245],
            5: [14.59, 10, 1.271, 1.784, 0.4753, 0.3053, 0.8918, 1.781, 5, 1.124, 0.01231, 3.088e-07, 0.0005655],
            6: [12.15, 9.994, 1.167, 1.826, 0.4596, 0.2525, 0.7496, 1.813, 4.999, 1.07, 0.01449, 4.216e-07, 0.0005457],
            7: [12.36, 9.996, 1.828, 1.134, 0.3237, 0.4213, 0.8098, 1.794, 4.999, 1.181, 0.01152, 3.925e-07, 0.0006236],
            8: [13.01, 9.999, 2.194, 1.257, 0.3552, 0.5099, 1.15, 1.755, 5, 1.198, 0.01355, 3.388e-07, 0.0005902],
            9: [12.84, 10, 1.321, 1.855, 0.4989, 0.314, 0.9809, 1.768, 4.999, 1.065, 0.0145, 3.573e-07, 0.0005508],
            10: [12.49, 9.997, 1.538, 1.768, 0.4931, 0.3753, 1.156, 1.765, 4.999, 1.132, 0.0152, 3.552e-07, 0.000621],
            11: [15.23, 10, 1.78, 1.549, 0.4475, 0.4129, 1.163, 1.765, 4.999, 1.2, 0.01337, 2.689e-07, 0.0006528],
        },
    },
    'BVI_S12': {
        'ULV_spheric_asym_C4': { # chi2 = 86 to 91
            1: [18.5, 9.581, 2.253, 1.013, 0.2621, 0.4879, 0.813, 1.793, 0.0001143, 0.0005801],
            2: [19.42, 8.5, 1.4, 1.4, 0.3983, 0.3436, 0.7708, 1.807, 0.000112, 0.0007232],
            3: [19.15, 8.5, 1.187, 1.891, 0.4825, 0.2699, 0.8141, 1.793, 0.0001127, 0.0005932],
            4: [19.46, 8.5, 1.852, 1.2, 0.3384, 0.4398, 0.8771, 1.784, 0.0001054, 0.0006845],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.95, 0.6849, 1.213, 1.934, 0.4786, 0.2423, 0.7785, 1.806, 3.15e-07, 0.0005697], # chi2 = 83
        },
        'ULV_sph_asym_prot_layer': {
            1: [14.4, 10, 1.189, 1.762, 0.4443, 0.2725, 0.7573, 1.808, 5, 1.2, 0.01328, 3.411e-07, 0.0006192], # PREMIUM chi2 = 72
            2: [14.77, 10, 1.499, 1.765, 0.4752, 0.3735, 1.106, 1.762, 4.992, 1.188, 0.01521, 2.918e-07, 0.0006728],
            3: [11.61, 9.992, 1.441, 1.877, 0.513, 0.3107, 1.047, 1.762, 5, 1.124, 0.01305, 4.056e-07, 0.0005962],
            4: [11.5, 9.993, 1.242, 1.981, 0.4615, 0.2163, 0.7408, 1.814, 5, 1.194, 0.01174, 4.585e-07, 0.0005608],
            5: [15.5, 10, 1.145, 1.855, 0.4512, 0.2572, 0.7448, 1.812, 5, 1.195, 0.01269, 3.11e-07, 0.0005947],
            6: [13.22, 9.951, 1.17, 1.839, 0.4527, 0.2565, 0.7503, 1.811, 4.982, 1.158, 0.01406, 3.859e-07, 0.0005948],
            7: [12.35, 9.999, 1.839, 1.14, 0.3301, 0.4171, 0.8153, 1.79, 4.999, 1.103, 0.0127, 3.974e-07, 0.0006816],
            8: [13.69, 9.999, 2.209, 1.254, 0.3588, 0.5065, 1.151, 1.752, 5, 1.199, 0.01411, 3.222e-07, 0.0006486],
            9: [12.63, 9.998, 1.323, 1.862, 0.4957, 0.3153, 0.9793, 1.766, 4.999, 1.091, 0.01539, 3.715e-07, 0.0006053],
            10: [15.06, 10, 1.602, 1.748, 0.4973, 0.3557, 1.14, 1.762, 5, 1.184, 0.01341, 2.833e-07, 0.0006636],
            11: [15.03, 10, 1.829, 1.505, 0.434, 0.4164, 1.141, 1.763, 4.999, 1.199, 0.01379, 2.788e-07, 0.0007057],
        },
    },
    'BVI_S13': {
        'ULV_spheric_asym_C4': { # chi2 = 1107
            1: [18.83, 9.104, 0.941, 0.8043, 0.4666, 0.4663, 0.5792, 1.81, 0.0001619, 7.294e-43],
            2: [19.09, 8.499, 0.9427, 0.8036, 0.4662, 0.4659, 0.5788, 1.81, 0.0001622, 2.862e-41],
            3: [19.09, 8.498, 0.9427, 0.8036, 0.4662, 0.4659, 0.5788, 1.81, 0.0001622, 1.109e-41],
            4: [19.09, 8.499, 0.9427, 0.8036, 0.4662, 0.4659, 0.5788, 1.81, 0.0001622, 1.899e-43],
            5: [19.08, 8.499, 0.9427, 0.804, 0.4662, 0.4658, 0.5789, 1.81, 0.0001623, 4.395e-44],
            6: [19.09, 8.498, 0.9427, 0.8036, 0.4662, 0.4659, 0.5788, 1.81, 0.0001622, 1.419e-44],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [12.64, 0.6503, 0.9434, 0.804, 0.4674, 0.4658, 0.5794, 1.81, 4.219e-07, 7.836e-42], # chi2 = 1107
        },
        'ULV_sph_asym_prot_layer': { # chi2 = 1124
            1: [14.17, 9.766, 1.01, 0.8551, 0.5053, 0.5028, 0.7, 1.743, 4.993, 1.007, 0.009379, 4.493e-07, 1.237e-35],
            2: [13.24, 9.955, 1.011, 0.847, 0.5083, 0.5027, 0.7, 1.74, 4.966, 1.08, 0.009473, 4.885e-07, 2.323e-37],
        },
    },
    'BVI_S14': {
        'ULV_spheric_asym_C4': {
            1: [19.67, 8.907, 1.139, 1.908, 0.509, 0.2972, 0.8503, 1.763, 0.0001031, 0.0007643],
            # not acceptable
        },
        "ULV_spheric_asym_C4_lognormal": {
            2: [9, 0.7451, 1.168, 1.723, 0.4556, 0.3019, 0.7631, 1.789, 4.515e-07, 0.0008046], 
            1: [12.5, 0.6281, 1.171, 1.711, 0.4608, 0.305, 0.7663, 1.791, 3.099e-07, 0.0008186],
            3: [10.7, 0.6812, 1.147, 1.901, 0.5025, 0.2964, 0.8429, 1.769, 3.625e-07, 0.0007566],
            4: [12, 0.6406, 1.157, 1.921, 0.5023, 0.2949, 0.8478, 1.77, 3.152e-07, 0.0007488],
            # 2: [9.002, 0.7429, 1.413, 1.369, 0.3691, 0.3732, 0.7448, 1.794, 4.562e-07, 0.000884],
        },
        'ULV_sph_asym_prot_layer': { # mostly chi2 between 80 and 90
            1: [11.98, 10, 1.124, 1.792, 0.4615, 0.2795, 0.7536, 1.788, 5, 1.2, 0.02147, 4.287e-07, 0.0007679],
            2: [12.76, 10, 1.418, 1.825, 0.4927, 0.3737, 1.102, 1.741, 4.998, 1.199, 0.02402, 3.552e-07, 0.0008219],
            3: [12.61, 10, 1.56, 1.873, 0.5465, 0.3254, 1.155, 1.738, 5, 1.2, 0.01748, 3.644e-07, 0.0007726],
            4: [12.91, 10, 1.678, 1.122, 0.359, 0.3538, 0.7091, 1.797, 5, 1.2, 0.01568, 3.933e-07, 0.0008723],
            5: [14.94, 9.995, 1.09, 1.884, 0.4698, 0.2718, 0.7586, 1.786, 4.982, 1.198, 0.0204, 3.205e-07, 0.0007406],
            6: [12.92, 10, 1.978, 1.018, 0.3259, 0.4276, 0.7819, 1.773, 5, 1.199, 0.01552, 3.736e-07, 0.0008211],
            7: [13.1, 10, 2.227, 1.252, 0.3829, 0.4999, 1.169, 1.733, 5, 1.2, 0.01875, 3.333e-07, 0.0008251],
            8: [11.3, 9.998, 1.289, 1.863, 0.4969, 0.33, 0.9759, 1.746, 4.998, 1.186, 0.02208, 4.206e-07, 0.0007702],
            9: [12.3, 9.998, 1.633, 1.788, 0.5292, 0.3485, 1.18, 1.742, 4.991, 1.176, 0.01889, 3.702e-07, 0.0008109],
            10: [13.48, 10, 1.954, 1.46, 0.4428, 0.4306, 1.183, 1.744, 4.998, 1.2, 0.01943, 3.167e-07, 0.0008698],
        },
    },
    'BVI_S15': {
        'ULV_spheric_asym_C4': { # all hard to accept 
            1: [21.11, 9.13, 1.266, 2.188, 0.5551, 0.2327, 0.8996, 1.748, 0.0001054, 0.0006566],
            2: [19.51, 8.5, 1.2, 1.829, 0.4867, 0.3149, 0.8559, 1.76, 0.0001192, 0.0008003],
            3: [19.27, 8.5, 2.5, 1.104, 0.2363, 0.55, 0.8995, 1.749, 0.0001251, 0.0006623],  
        },
        "ULV_spheric_asym_C4_lognormal": {
            2: [9.823, 0.7317, 1.173, 1.855, 0.4859, 0.3039, 0.8328, 1.767, 4.383e-07, 0.0007747], # acceptable
            1: [12.5, 0.6472, 1.167, 1.728, 0.4588, 0.3039, 0.7605, 1.787, 3.435e-07, 0.000824],
            3: [10.7, 0.6943, 1.168, 1.893, 0.4965, 0.3022, 0.8485, 1.763, 4.016e-07, 0.0007641],
            4: [12, 0.653, 1.174, 1.894, 0.5005, 0.305, 0.8596, 1.762, 3.51e-07, 0.0007697],
        },
        'ULV_sph_asym_prot_layer': {
            1: [10.53, 10, 1.503, 1.876, 0.5283, 0.3013, 1.045, 1.739, 5, 1.081, 0.01614, 5.117e-07, 0.0007531], # PREMIUM, though parameters are hard to interpret ...
            2: [10.53, 10, 1.309, 1.965, 0.4733, 0.209, 0.7315, 1.793, 5, 1.158, 0.0143, 5.668e-07, 0.0007331], 
        }
    },
    'BVI_S16': {
        'ULV_spheric_asym_C4': {
            1: [17.1, 9.5, 1.353, 2.459, 0.6214, 0.2259, 1.047, 1.717, 0.000138, 0.00062],
            2: [15.16, 9.5, 2.869, 1.153, 0.2272, 0.6198, 1.044, 1.718, 0.0001646, 0.0006234],

        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [22.11, 0.4393, 1.107, 2.086, 0.5166, 0.3134, 0.8884, 1.75, 1.424e-07, 0.000807], 
        },
        'ULV_sph_asym_prot_layer': {
            1: [12.52, 10, 1.156, 2.038, 0.511, 0.3145, 0.9102, 1.747, 4.845, 0.9185, 0.01735, 3.894e-07, 0.0007986], # PREMIUM, though parameters are hard to interpret ...
        }
    },
    'BVI_S17': { # very different chi2, but no big differences in the curves ???
        'ULV_spheric_asym_C4': { 
            1: [11.53, 9.732, 1.415, 2.318, 0.5605, 0.3738, 1.2, 1.73, 0.0001412, 0.00073],
            2: [11.12, 9.642, 0.9951, 2.414, 0.4904, 0.2506, 0.6695, 1.802, 0.000171, 0.0006121],
            3: [11.27, 9.689, 2.929, 1.295, 0.2544, 0.6689, 1.2, 1.708, 0.0001775, 0.0005493],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [20.57, 0.4049, 1.046, 2.314, 0.5101, 0.3031, 0.7948, 1.77, 1.199e-07, 0.0006632], # best acceptable chi2 = 162
        },
        'ULV_sph_asym_prot_layer': {
            1: [15.57, 9.998, 1.066, 2.19, 0.4751, 0.3019, 0.7368, 1.785, 2.301, 0.7237, 2.882e-05, 2.095e-07, 0.00069], # chi2 = 295
            2: [13.6, 10, 1.184, 2.141, 0.4953, 0.3445, 0.8803, 1.752, 3.749, 0.9006, 1.367e-05, 2.396e-07, 0.0007334],
            3: [14.93, 9.85, 1.274, 2.148, 0.4827, 0.3609, 0.9408, 1.747, 2.737, 0.7005, 1.182e-05, 2.069e-07, 0.0007415],
            4: [16.02, 9.992, 2.5, 1.297, 0.3991, 0.4978, 1.146, 1.736, 2.408, 1.007, 3.825e-05, 1.796e-07, 0.0007845],
        },
    },
    'BVI_S18': {
        'ULV_spheric_asym_C4': {
            1: [19.01, 8.961, 2.459, 1.084, 0.2269, 0.5257, 0.8355, 1.768, 0.0001065, 0.0006244], 
            # not acceptable
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [12.68, 0.6141, 1.161, 1.938, 0.5017, 0.2937, 0.8503, 1.768, 2.856e-07, 0.0006721], # chi2 = 104
        },
        'ULV_sph_asym_prot_layer': {
            1: [10.42, 10, 1.117, 1.87, 0.4603, 0.2607, 0.732, 1.795, 4.999, 1.125, 0.01982, 4.838e-07, 0.0006739], # best acceptable chi2 = 79
            2: [10.91, 9.842, 1.902, 1.105, 0.3443, 0.4137, 0.809, 1.771, 4.998, 0.948, 0.01685, 4.343e-07, 0.0007637], # chi2 = 87
            3: [11.93, 10, 2.277, 1.232, 0.3689, 0.5055, 1.151, 1.732, 5, 1.179, 0.01708, 3.585e-07, 0.0007326], # chi2 = 81
        },
    },
    'BVI_S19': {
        'ULV_spheric_asym_C4': { # all pretty bad
            1: [19.22, 9.523, 2.408, 0.9622, 0.2657, 0.5112, 0.8358, 1.775, 0.0001022, 0.0005281],
            2: [10.33, 10, 1.058, 3.744, 0.5604, 0.13, 0.7894, 1.787, 0.0002374, 0.0002321],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [12, 0.6622, 1.168, 2.01, 0.5088, 0.2862, 0.8794, 1.771, 2.951e-07, 0.0005472], # chi2 = 89
            2: [10.7, 0.7007, 1.182, 1.906, 0.4844, 0.2911, 0.8403, 1.779, 3.405e-07, 0.0005757], # chi2 = 88
            3: [9.623, 0.7453, 1.141, 2.046, 0.5054, 0.2741, 0.8523, 1.774, 3.791e-07, 0.0005243], # 2nd best
        },
        'ULV_sph_asym_prot_layer': {
            1: [11.18, 9.998, 1.302, 1.896, 0.4962, 0.3201, 0.9771, 1.757, 4.999, 1.13, 0.01907, 4.118e-07, 0.0005848], # best acceptable
            2: [10.38, 9.998, 1.848, 1.482, 0.4278, 0.407, 1.094, 1.754, 4.557, 1.033, 0.01763, 4.21e-07, 0.0006758],
            3: [10.96, 9.945, 1.334, 1.466, 0.3881, 0.3271, 0.7352, 1.805, 4.999, 1.059, 0.02013, 4.65e-07, 0.0006759],
        }
    },
    'BVI_S20': {
        'ULV_spheric_asym_C4': { # all pretty bad
            1: [8.015, 6.543, 1.037, 3.637, 0.6202, 0.13, 0.8107, 1.767, 0.0005009, 0.0003387],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [9.001, 0.6941, 1.217, 1.724, 0.4691, 0.3057, 0.8049, 1.781, 5.008e-07, 0.0007936], # best chi2 = 94
            2: [9, 0.6918, 1.455, 1.383, 0.3869, 0.3745, 0.7878, 1.786, 5.03e-07, 0.0008742], # 2nd best
            3: [10.7, 0.6499, 1.17, 1.911, 0.5118, 0.2879, 0.8511, 1.771, 3.939e-07, 0.0007384], # chi2 = 109
        },
        'ULV_sph_asym_prot_layer': {
            1: [15.42, 10, 1.872, 1.541, 0.4645, 0.4092, 1.196, 1.746, 4.999, 1.2, 0.02394, 2.71e-07, 0.0008663], # 3rd best
            2: [10.37, 9.999, 1.645, 1.752, 0.517, 0.3579, 1.176, 1.745, 4.998, 1.151, 0.02381, 4.561e-07, 0.0008196],
        },
    }, 
    'BVII_S1': {     
        'ULV_spheric_asym_C4': { 
            1: [8.012, 5.757, 1.309, 3.5, 0.5706, 0.1497, 0.8063, 1.882, 0.0005653, 9.016e-05],
            2: [9, 5.565, 1.375, 2.5, 0.5489, 0.2125, 0.8162, 1.882, 0.0004858, 0.0003475],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.41, 0.4883, 2.544, 1.131, 0.2737, 0.4705, 0.7712, 1.885, 5.183e-07, 0.0005526],
        },
    },
    'BVII_S2': {     
        'ULV_spheric_asym_C4': { 
            1: [8, 5.046, 1.388, 3.5, 0.565, 0.1454, 0.8013, 1.853, 0.0006322, 0.0002682],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.3663, 2.6, 1.123, 0.3248, 0.5286, 0.9228, 1.843, 6.159e-07, 0.0007876],
        },
    },
    'BVII_S3': {     
        'ULV_spheric_asym_C4': { 
            1: [8.164, 5.443, 3.372, 1.058, 0.3583, 0.4753, 1.073, 1.785, 0.0001198, 0.0004488],
            2: [9, 5.248, 2.327, 1.666, 0.4237, 0.2105, 0.747, 1.837, 0.0001246, 0.000478],
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
    },
    'BVII_S4': { # bgr sub removed too much of the curve    
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S5': {     
        'ULV_spheric_asym_C4': { 
            # 1: [8.375, 6.267, 0.9268, 1.738, 0.277, 0.2665, 0.6375, 1.815, 0.0004056, 0.0006709],
            1: [18.85, 6.191, 1.6, 1.6, 0.3674, 0.1284, 0.6484, 1.812, 0.0001153, 0.0007696],
            2: [9, 6.11, 1.4, 1.4, 0.1869, 0.3348, 0.6469, 1.813, 0.0003781, 0.0007477],
            3: [19.2, 6.119, 1.6, 1.083, 0.3777, 0.2, 0.6742, 1.801, 0.0001095, 0.0007979],
            4: [15.35, 7.075, 2.113, 1, 0.389, 0.3476, 1, 1.737, 0.0001244, 0.0007139],
            5: [18.43, 6.202, 2.186, 0.722, 0.2673, 0.2758, 0.6371, 1.815, 0.0001172, 0.0006718],
            6: [9, 6.286, 1.089, 1.856, 0.407, 0.3534, 1, 1.729, 0.0002953, 0.0006795],
            7: [11, 6.575, 1.043, 1.6, 0.2419, 0.2974, 0.644, 1.813, 0.0002701, 0.0007123],
            8: [13.96, 8.107, 1.564, 1.414, 0.3858, 0.15, 0.6677, 1.804, 0.0001726, 0.0007859],
            9: [18.38, 6.215, 1.898, 1.2, 0.4159, 0.275, 0.9553, 1.747, 9.62e-05, 0.0007179],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [20.24, 0.2822, 2.127, 0.8, 0.2684, 0.2496, 0.6239, 1.825, 1.87e-07, 0.0006777],
            2: [9.001, 0.5436, 2.026, 1.312, 0.4541, 0.3407, 1.2, 1.746, 4.843e-07, 0.0007237],
        },
    },
    'BVII_S6': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S7': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S8': {     
        'ULV_spheric_asym_C4': { 
            1: [8, 5.727, 1.307, 3.5, 0.5696, 0.1486, 0.7985, 1.885, 0.0005767, 0.0002766],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.41, 0.4707, 2.565, 1.169, 0.2696, 0.4675, 0.7761, 1.889, 5.344e-07, 0.000725],
        },
    },
    'BVII_S9': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
    },
    'BVII_S10': {     
        'ULV_spheric_asym_C4': { 
            1: [18.84, 6.09, 1.878, 1.2, 0.3127, 0.1546, 0.6235, 1.817, 9.774e-05, 0.0007237],
            2: [11, 6.413, 1.095, 1.6, 0.2174, 0.3018, 0.6367, 1.812, 0.000232, 0.0007168],
            3: [8.617, 6.093, 0.9452, 1.732, 0.2555, 0.2719, 0.6306, 1.814, 0.0003376, 0.000683],
            4: [9, 5.963, 1.4, 1.4, 0.1781, 0.3412, 0.644, 1.809, 0.0003254, 0.0007551],
            5: [15.04, 7.222, 2.01, 1.2, 0.4174, 0.2805, 1, 1.739, 0.0001076, 0.0007098],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [20.25, 0.2771, 2.06, 0.8, 0.2815, 0.2355, 0.6169, 1.823, 1.601e-07, 0.0007063],
            2: [11.88, 0.4659, 0.8663, 1.743, 0.2707, 0.2856, 0.6378, 1.817, 3.432e-07, 0.0007014],
        },
    },
    'BVII_S11': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S12': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S13': {     
        'ULV_spheric_asym_C4': { 
            1: [8, 5.302, 1.323, 3.5, 0.5665, 0.1473, 0.788, 1.867, 0.0005886, 0.0002136],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.3879, 2.6, 1.139, 0.3142, 0.5243, 0.9155, 1.844, 5.7e-07, 0.0006994],
        },
    },
    'BVII_S14': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S15': {     
        'ULV_spheric_asym_C4': { 
            1: [15.94, 6.352, 2.087, 0.7394, 0.2934, 0.3227, 0.708, 1.793, 0.0001445, 0.000684],
            2: [16.07, 6.32, 1.784, 1, 0.3388, 0.2346, 0.6975, 1.799, 0.0001453, 0.0007277],
            3: [11, 7.584, 1.4, 1.375, 0.2055, 0.3512, 0.6898, 1.802, 0.0002468, 0.0007354],
            4: [19.37, 7.442, 2.196, 0.9009, 0.3548, 0.4072, 1, 1.734, 8.554e-05, 0.0006834],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [18.49, 0.2958, 1.925, 0.8, 0.3048, 0.2934, 0.6825, 1.803, 2.157e-07, 0.0007185],
        },
    },
    'BVII_S16': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S17': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S18': {     
        'ULV_spheric_asym_C4': { 
            1: [8, 5.425, 1.297, 3.5, 0.5808, 0.1482, 0.7962, 1.873, 0.0005955, 0.000271],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.4079, 2.599, 1.073, 0.2899, 0.4794, 0.7818, 1.885, 5.882e-07, 0.0007293],
        },
    },
    'BVII_S19': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S20': {     
        'ULV_spheric_asym_C4': { 
            1: [16.64, 5.808, 1.814, 0.8462, 0.3247, 0.2835, 0.6929, 1.795, 0.0001526, 0.0007656],
            2: [13.54, 6.672, 1.6, 1.173, 0.3664, 0.2, 0.6822, 1.801, 0.0002129, 0.0007852],
            3: [14.67, 6.336, 2.004, 1, 0.3909, 0.3694, 1, 1.734, 0.0001578, 0.0007581],
            4: [11, 7.932, 1.153, 1.4, 0.2558, 0.3385, 0.6884, 1.798, 0.0002605, 0.0007761],
            5: [17.49, 5.639, 1.709, 1.2, 0.4167, 0.2694, 0.89, 1.752, 0.0001243, 0.0007587],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [17.96, 0.2928, 1.916, 0.8, 0.3042, 0.3064, 0.6915, 1.798, 2.466e-07, 0.0007385],
        },
    },
    'BVII_S21': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BVII_S22': {     
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
        "ULV_sph_asym_prot_layer": {
        },
    },
    'BIX_S1': {    # 15 mM POPG , all fits roughly equally good
        'ULV_spheric_asym_C4': {
            1: [21.18, 9.748, 3.9, 0.9176, 0.1362, 0.5116, 0.755, 1.819, 9.272e-05, 0.0002047], # best
            2: [21.19, 9.777, 1.056, 3.539, 0.5129, 0.13, 0.7548, 1.819, 9.26e-05, 0.0001923],
            3: [22.31, 8.499, 1.4, 1.4, 0.3935, 0.3401, 0.779, 1.815, 8.608e-05, 0.0005684],
            4: [21.54, 8.5, 1.102, 2.5, 0.4997, 0.1869, 0.7701, 1.814, 9.305e-05, 0.0003135],
            5: [21.7, 8.5, 2.5, 0.987, 0.2212, 0.4912, 0.7898, 1.809, 9.071e-05, 0.0003785],
            6: [22.27, 8.5, 2, 1.2, 0.3196, 0.4681, 0.9432, 1.783, 8.021e-05, 0.0004965],
            7: [19.6, 8.483, 1.2, 2.046, 0.4937, 0.2396, 0.8293, 1.8, 0.0001058, 0.000397],
        },
        "ULV_spheric_asym_C4_lognormal": {
            2: [10.86, 0.7143, 3, 0.9518, 0.1812, 0.4986, 0.7707, 1.81, 3.479e-07, 0.0003053], 
            1: [10.79, 0.7079, 1.104, 2.5, 0.4985, 0.1854, 0.7654, 1.821, 3.566e-07, 0.0003044],
        },
        "ULV_sph_asym_prot_layer": {
            1: [17.94, 10, 1.167, 1.992, 0.4605, 0.2272, 0.7503, 1.821, 5, 1.2, 0.008231, 2.435e-07, 0.0004234], # as good as best
            2: [18.47, 10, 1.524, 1.753, 0.4806, 0.3625, 1.114, 1.771, 4.998, 1.167, 0.01062, 2.038e-07, 0.0005249],
        },
    },
    'BIX_S2': {      # 15 mM DOPC/GM1 (80/20)
        'ULV_spheric_asym_C4': { # ALL SPH ASYM QUITE BAD
            # 1: [10, 7.824, 0.7555, 2.04, 0.9219, 0.1588, 0.8167, 1.746, 0.0002865, 0.0004348], 
            # 2: [9, 7.58, 0.9218, 1.4, 0.8, 0.2499, 0.8732, 1.757, 0.0002916, 0.0005339],
            # 3: [9, 7.307, 1.2, 1.2, 0.688, 0.339, 1, 1.77, 0.0002549, 0.0005917],
            # 4: [11, 6.564, 0.9252, 1.679, 0.8, 0.2, 0.8507, 1.775, 0.0002418, 0.0004923],
        },
        "ULV_sph_asym_prot_layer": {
            1: [9, 7.948, 1.094, 1.183, 0.3866, 0.299, 0.7474, 1.749, 3.008, 0.7865, 0.185, 5.086e-07, 0.0005352], # chi2 = 95
        },
        "ULV_lognormal_sph_asym_prot_layer": { # all quite equally good
            1: [11.55, 0.497, 1.546, 0.8, 0.2563, 0.3353, 0.718, 1.757, 2.517, 0.9936, 0.231, 3.412e-07, 0.0005203], # chi2 = 59 BEST
            2: [13, 0.5003, 1.004, 1.369, 0.4061, 0.1925, 0.7809, 1.757, 2.163, 1.2, 0.2349, 2.849e-07, 0.0005142], # chi2 = 67
            3: [15, 0.5432, 1.137, 1.115, 0.3412, 0.3229, 0.724, 1.759, 3, 0.7787, 0.1947, 1.87e-07, 0.0005393], # chi2 = 84
        },
        "ULV_lognormal_sph_asym_TWO_prot_layers": {
            1: [11.76, 0.4971, 1.419, 1.047, 0.2309, 0.3814, 0.7298, 1.747, 2.872, 0.7839, 0.1602, 2.522, 0.7008, 0.08372, 3.133e-07, 0.0005331],
        }
    },
    'BIX_S10': {     
        'ULV_spheric_asym_C4': { 
            1: [22.31, 8.5, 1.4, 1.4, 0.3882, 0.3513, 0.6805, 1.943, 0.0001066, 0.0007931],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.6121, 2.589, 1.163, 0.2387, 0.4481, 0.752, 1.937, 4.043e-07, 0.000481],
        },
    },
    'BIX_S11': {     
        'ULV_spheric_asym_C4': { 
            1: [21.82, 8.5, 1.87, 1.404, 0.3489, 0.3972, 0.8131, 1.923, 8.858e-05, 0.0006424],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.6048, 2.599, 1.162, 0.2426, 0.4561, 0.7663, 1.931, 3.893e-07, 0.0004718],
        },
    },
    'BIX_S12': {     
        'ULV_spheric_asym_C4': { 
            1: [21.92, 8.5, 1.941, 1.383, 0.3472, 0.4183, 0.845, 1.911, 8.969e-05, 0.0007212],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.6006, 2.599, 1.16, 0.248, 0.4674, 0.7827, 1.922, 4.03e-07, 0.0005525],
        },
    },
    'BIX_S13': {     
        'ULV_spheric_asym_C4': { 
            1: [21.46, 8.5, 1.47, 1.871, 0.456, 0.3045, 0.8255, 1.91, 9.733e-05, 0.000621],
            2: [21.44, 8.5, 2.182, 1.258, 0.3057, 0.4555, 0.8259, 1.91, 9.743e-05, 0.0006217],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.5852, 2.599, 1.157, 0.2531, 0.4753, 0.7924, 1.914, 4.241e-07, 0.000506],
        },
    },
    'BIX_S14': {     
        'ULV_spheric_asym_C4': { 
            1: [21, 9.132, 2.371, 1.213, 0.2911, 0.4865, 0.853, 1.894, 0.000112, 0.0007839],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [11.4, 0.5601, 2.6, 1.155, 0.2652, 0.4938, 0.8284, 1.894, 5.01e-07, 0.000715],
        },
    },
    'BIX_S15': {     #: "15 mM DOPC/DOPS (8/2) + 6 μM aSynWT",
        'ULV_spheric_asym_C4': { 
            1: [17.06, 2.762, 1.538, 1.006, 0.3722, 0.3436, 0.8049, 1.778, 0.0001357, 0.0005374],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [16.65, 0.1557, 1.458, 1.092, 0.3929, 0.3237, 0.8129, 1.776, 2.803e-07, 0.0005438],
        },
    },
    'BIX_S16': {     
        'ULV_spheric_asym_C4': { 
            1: [17.01, 2.854, 1.528, 1.034, 0.3769, 0.3362, 0.8107, 1.774, 0.0001385, 0.0005207],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [16.47, 0.1644, 1.52, 1.06, 0.3793, 0.3364, 0.82, 1.772, 2.887e-07, 0.0005155],
        },
    },
    'BIX_S17': {     
        'ULV_spheric_asym_C4': { 
            1: [16.97, 3.06, 1.481, 1.052, 0.3759, 0.3049, 0.7662, 1.784, 0.0001407, 0.000602],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [16.47, 0.1729, 1.532, 1.028, 0.3653, 0.3199, 0.7749, 1.783, 2.886e-07, 0.0006015],
        },
    },
    'BIX_S18': {     
        'ULV_spheric_asym_C4': { 
            1: [16.58, 4.304, 1.74, 0.8465, 0.3303, 0.3479, 0.751, 1.785, 0.000144, 0.0006345],
            2: [16.63, 4.292, 1.4, 1.264, 0.3956, 0.2214, 0.718, 1.798, 0.0001488, 0.0006404],
            # 3: [8, 4.981, 1.142, 1.352, 0.3632, 0.3353, 0.7796, 1.778, 0.0004647, 0.000633],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [16.54, 0.244, 1.805, 0.8, 0.3092, 0.3633, 0.729, 1.794, 2.781e-07, 0.0006374],
        },
    },
    'BIX_S19': {     
        'ULV_spheric_asym_C4': { 
            1: [17.2, 5.494, 1.674, 1, 0.3456, 0.2429, 0.6897, 1.801, 0.0001375, 0.0007067],
            3: [17.07, 5.518, 1.808, 1.2, 0.4262, 0.3127, 1, 1.741, 0.0001179, 0.0006885],
            4: [17.6, 5.476, 1.4, 1.4, 0.4024, 0.1697, 0.6719, 1.805, 0.0001388, 0.0007463],
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [18.42, 0.2728, 1.82, 0.8333, 0.3201, 0.3054, 0.7023, 1.794, 2.239e-07, 0.0006907],
        },
    },
    'BXII_S1': {     
        'ULV_spheric_asym_C4': { 
            1: [15.03, 5.755, 1.288, 1.016, 0.3092, 0.3289, 0.6363, 1.919, 0.0001761, 0.0004419], # chi2 = 50, excellent fit
        },
        "ULV_spheric_asym_C4_lognormal": {
            1: [17.14, 0.3129, 1.258, 1.059, 0.336, 0.3079, 0.6488, 1.916, 2.527e-07, 0.0004371],
        },
    },
    'BXII_S3': {     
        'ULV_spheric_asym_C4': { 
            1: [14.85, 3.385, 1.574, 0.7849, 0.3357, 0.4043, 0.7006, 1.885, 0.0001613, 0.0004424], # chi2 = 137
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
    },
    'BXII_S4': {     
        'ULV_spheric_asym_C4': { 
            1: [15.12, 3.266, 1.75, 0.7189, 0.3201, 0.4592, 0.7362, 1.86, 0.0001587, 0.0004577], # chi2 = 195
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
    },
    'BXII_S6': {     # 15 mM DOPC, curve points towards multilamellarity, fusion
        'ULV_spheric_asym_C4': { 
        },
        "ULV_spheric_asym_C4_lognormal": {
        },
    },
    "BXIII_S1": { # "23nM SVs",
    },
    "BXIII_S2": { # "CSLBs",
        "hard_sphere_polydisp": {
            1: [26.91, 1.662, 7.474e-09, 0.001673], # pretty bad
        },
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.19, 0.5834, 1.187, 0.0228, 7.52e-09, 0.001064, 0.00154, 1],
            1: [27.19, 0.4084, 0.9592, 0.1784, 7.098e-09, 0.001004, 0.003202, 1.203],
        },
    },
    "BXIII_S3": { # "3 μM Syn1",
    },
    "BXIII_S4": { # "15mM LV4 + 4μM Syn1",
    },
    "BXIII_S5": { # "15mM LV4 + 6μM Syn1",
    },
    "BXIII_S6": { # "15mM LV4 + 6μM Syn1 + 24μM aSynWT",
    },
    "BXIII_S11": { # "CSLB + 6μM Syn1",
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.32, 0.8808, 1.2, 0.08813, 3.352e-09, 0.0002287, 0.0181, 1.4],
        },
        "ellips_DAB_struc_factor": {
            1: [3.44, 18.65, 3.108, 0.1199, 1.623, 0.1221, 0.0388],
        },
    },
    "BXIII_S12": { # "CSLB + 4μM Syn1",
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.34, 0.6142, 1.2, 0.05759, 3.017e-09, 0.0005999, 0.01091, 1.4],
        },
        "ellips_DAB_struc_factor": {
            1: [4.234, 3.923, 13.37, 0.3838, 9, 0.15, 0.3154],
        },
    },
    "BXIII_S13": { # "CSLB + 6μM Syn1 + 12μM aSynWT",
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.34, 1.111, 1.2, 0.09494, 3.477e-09, 0.0002991, 0.02233, 1.4],
        },
        "ellips_DAB_struc_factor": {
            1: [3.207, 18.23, 3.053, 0.1205, 1.477, 0.1212, 0.03941],
        },
    },
    "BXIII_S14": { # "CSLB + 6μM Syn1 + 24μM aSynWT",
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.38, 0.8424, 1.2, 0.07788, 4.537e-09, 0.0005131, 0.02226, 1.4],
        },
        "ellips_DAB_struc_factor": {
            1: [2.363, 100, 1.248, 0.05707, 7.096e-05, 0.005395, 0.0006818],
        },
    },
    "BXIV_S1": { # "15mM LV4 + 6μM Syn1 + 24μM ATP",
    },
    "BXIV_S2": { # "23nM SVs + 6μM Syn1 + 24μM ATP",
    },
    "BXIV_S3": { # "CSLB + 6μM Syn1 + 24 μM ATP"
        "ellipsoid_homogen_DAB": { # negative means q > 0.25 /nm
            -1: [26.37, 1.034, 1.2, 0.09486, 3.212e-09, 0.0004608, 0.01256, 1.4],
        },
        "ellips_DAB_struc_factor": {
            1: [3.975, 3.909, 13.24, 0.3853, 9, 0.15, 0.3174],
        },
    }
}


