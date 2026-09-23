import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
# import fabio
# import pyFAI
import os
from scipy import optimize
from scipy.optimize import curve_fit
import scipy
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

plt.rcParams['font.size']   = 18
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac, amsmath}'

# =========== Guinier fit =====================================
# remember: q * Rg < 1.3 for the Guinier approximation to be valid
def guinier_fit(q, I0, Rg):
    return I0 * np.exp(-q**2 * Rg**2 / 3)

# ========= most models come with this power law background ==========================
def powerlaw_bgr(q, c2, c3, c4):
    return c2 * q**(-c3) + c4

base_path = Path("/home/AG_Salditt/Projekte_Membranbiophysik/Riko_BA/Data/BM26/Scatt_curves_average_for_same_position/")


#============== DAB (Debye Anderson Brumberger) Model =====================================
def DAB_model(q, c, xi):
    return c / (1 + (q*xi)**2)**2

#=============== polydispersed ellipsoidal CSLB model ========================================================================
rho_c = -1
def CSLB_ellipsoidal(q, R, sigma_R, eps, sigma_eps, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2, scale, c4):
    integrate_CSLB_partial = partial(integrate_CSLB_ellips, R=R, sigma_R=sigma_R, eps=eps, sigma_eps=sigma_eps, z_h=z_h, 
                                     sigma_h=sigma_h, rho_h=rho_h, z_c=z_c, sigma_c=sigma_c, rho_SiO2=rho_SiO2)
    
    with Pool(processes=8) as pool:  # Adjust processes to your server's cores
        I_res = np.array(list(tqdm(pool.imap(integrate_CSLB_partial, q), total=len(q), desc="Computing I(q)", ncols=80)))
        # I_res = np.array(list(pool.imap(integrate_CSLB_partial, q)))

    return scale * I_res + c4

def F_ellipsoid_homogen(theta, q, R, eps):
    """
    Vectorized form factor of a homogeneous ellipsoid.
    theta can be an array.
    """
    u = q * R * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    volume = 4/3 * np.pi * eps * R**3
    return 3 * volume * (np.sin(u) - u * np.cos(u)) / u**3

def F_CSLB_ellips(theta, q, R, eps, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2):
    R_tot      = R + z_c
    hom_ellipsoid = F_ellipsoid_homogen(theta, q, R, eps) * rho_SiO2

    u  = q *  R_tot * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    up = q * (R_tot + z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    um = q * (R_tot - z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    # Precompute exponentials and trig
    exp_h = np.exp(-sigma_h**2 * u**2 / (2 * R_tot**2))
    exp_c = np.exp(-sigma_c**2 * u**2 / (2 * R_tot**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    
    term_h = (
        (R_tot + z_h)**2 * sin_u_Rp / up +
        (R_tot - z_h)**2 * sin_u_Rm / um +
        sigma_h**2 * (cos_u_Rp + cos_u_Rm)
    )
    term_c = R_tot**2 * sin_u_R / u + sigma_c**2 * cos_u_R
    
    res = (4 * np.pi * np.sqrt(2*np.pi) * eps *
           (rho_h * sigma_h * exp_h * term_h -
            rho_c * sigma_c * exp_c * term_c))

    return hom_ellipsoid + res

def integrate_CSLB_ellips(q, R, sigma_R, eps, sigma_eps, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2):
    
    n_R, n_eps = 40, 10
    R_vals   = np.linspace(R - 3*sigma_R, R + 3*sigma_R, n_R)
    eps_vals = np.linspace(eps - 3*sigma_eps, eps + 3*sigma_eps, n_eps)
    theta = np.linspace(0, np.pi/2, 250)  # fixed grid for integration
    sin_theta = np.sin(theta)
    
    I = 0
    for R_i in R_vals:
        for eps_j in eps_vals:
            F_vals = F_CSLB_ellips(theta, q, R_i, eps_j, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2)
            F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)
            I += P_R(R_i, R, sigma_R) * P_eps(eps_j, eps, sigma_eps) * F_int
    
    dR = R_vals[1] - R_vals[0]
    deps = eps_vals[1] - eps_vals[0]
    return I * dR * deps
# ============================================================================================ 
    
#============== Skewed Cauchy-Lorentz distribution ========================================

def I_CL(q, scale, sigma, mu, lam, m, b):
    return scale/(sigma * np.pi) * (1 + (q - mu)**2 / (sigma * (1 + lam * (q - mu)))**2)**(-1) + m * q + b

#================= rough/fuzzy polydispersed spheres model =================================

def rough_hard_sphere_polydisp(q, R, sigma_R, a, scale, c2, c3, c4):

    # Prepare arguments for parallel map
    args_list = [(q_val, R, sigma_R, a) for q_val in q]
    
    with Pool(processes=10) as pool:  # Adjust processes to your server's cores
        I_res = np.array(pool.map(integrate_rough_for_q, args_list))
    # threshold = 1e-8
    # for i in range(len(I_res)):
    #     if I_res[i] < threshold:
    #         I_res[i] = (I_res[i+1] + I_res[i-1])/2
    return scale * I_res + c2 * q**(-c3) + c4

def integrate_rough_for_q(args, n_pts=1000):
    q_val, R, sigma_R, a = args

    # Integration range: cover +5σ around mean radius
    r_min = max(0, R - 5 * sigma_R)
    r_max = R + 5 * sigma_R

    # Discretize radius axis
    r = np.linspace(r_min, r_max, n_pts)

    # Gaussian distribution
    gaussian = (1 / (sigma_R * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - R) / sigma_R)**2)

    # Volume and intensity kernel
    volume = (4/3) * np.pi * (r + 2/np.sqrt(a))**3

    kernel = gaussian * volume**2 * np.abs(hard_sphere_raw(q_val, r) * Debye_Waller_factor(q_val, a))**2

    # Integrate using the trapezoid rule
    I_integrated = np.trapezoid(kernel, r)

    return I_integrated

def Debye_Waller_factor(q, a):
    return np.exp(-q**2 * a)

#=============== polydispersed CSLB model ========================================================================
rho_c = -1
def CSLB_spherical(q, R, sigma_R, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2, scale, c4):
    integrate_CSLB_partial = partial(integrate_CSLB, R=R, sigma_R=sigma_R, z_h=z_h, 
                                     sigma_h=sigma_h, rho_h=rho_h, z_c=z_c, sigma_c=sigma_c, rho_SiO2=rho_SiO2)
    
    with Pool(processes=8) as pool:  # Adjust processes to your server's cores
        # I_res = np.array(list(tqdm(pool.imap(integrate_CSLB_partial, q), total=len(q), desc="Computing I(q)", ncols=80)))
        I_res = np.array(list(pool.imap(integrate_CSLB_partial, q)))

    return scale * I_res + c4

def F_CSLB(q, R, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2):
    R_tot      = R + z_c
    hom_sphere = hard_sphere_raw(q, R) * rho_SiO2
    
    rho   = np.array([rho_h, rho_c, rho_h])#[None, :]              # shape (1, 3)
    sigma = np.array([sigma_h, sigma_c, sigma_h])#[None, :]        # shape (1, 3)
    delta = np.array([R_tot - z_h, R_tot, R_tot + z_h])#[:, None]  # shape (1, 3)
    # print(f"shapes: {np.shape(rho)}, {np.shape(sigma)}, {np.shape(delta)} \n")
    res = np.sum(
        rho * sigma * np.exp(-(q * sigma)**2 * 0.5)
        * (delta * np.sin(q * delta) + sigma**2 * q * np.cos(q * delta))
    )
    return hom_sphere + 2/q * res

def integrate_CSLB(q, R, sigma_R, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2):
    
    n_R    = 20
    R_vals = np.linspace(R - 3*sigma_R, R + 3*sigma_R, n_R)
    dR = R_vals[1] - R_vals[0]

    F      = np.array([F_CSLB(q, R_vals_i, z_h, sigma_h, rho_h, z_c, sigma_c, rho_SiO2) for R_vals_i in R_vals]) # shape (20, )
    # print(f"F: {np.shape(F)}\n")
    volume = 4/3 * np.pi * (R_vals + z_c)**3
    I      = dR * np.sum(P_R(R_vals, R, sigma_R) * volume**2 * np.abs(F)**2)

    return I 

#============= ASYMMETRIC ellipsoid lipid bilayer vesicle model =============================================
rho_c = 1
from scipy.integrate import quad
def ULV_ellipsoid_asym(q, R, sigma_R, eps, sigma_eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c2, c3, c4):
    I_func_partial = partial(ell_asym_integrator, R=R, sigma_R=sigma_R, eps=eps,
                             sigma_eps=sigma_eps, rho_h1=rho_h1, rho_h2=rho_h2,
                             sigma_h1=sigma_h1, sigma_h2=sigma_h2, sigma_c=sigma_c, z_h=z_h)
    with Pool(processes=10) as pool:  # 6–8 is ideal on typical CPUs
        I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        #I_res = pool.map(I_func_partial, q)
    
    return scale * np.array(I_res) + c2 * q**(-c3) + c4

def ell_asym_integrator(q, R, sigma_R, eps, sigma_eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h):
    #q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h = args
    
    n_R, n_eps = 60, 10 #100, 100
    R_vals   = np.linspace(R   - 3*sigma_R,     R + 3*sigma_R,     n_R)
    eps_vals = np.linspace(eps - 3*sigma_eps, eps + 3*sigma_eps, n_eps)
    theta = np.linspace(0, np.pi/2, 250)  # fixed grid for integration
    sin_theta = np.sin(theta)
    
    I = 0
    for R_i in R_vals:
        for eps_j in eps_vals:
            F_vals = F_ell_asym(theta, q, R_i, eps_j, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h)
            F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)         
            I += P_R(R_i, R, sigma_R) * F_int * P_eps(eps_j, eps, sigma_eps)
    
    dR = R_vals[1] - R_vals[0]
    deps = eps_vals[1] - eps_vals[0]
    return I * dR * deps

def F_ell_asym(theta, q, R, eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h):
    u  = q *  R        * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    up = q * (R + z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    um = q * (R - z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    # Precompute exponentials and trig
    exp_h2 = np.exp(-sigma_h2**2 * u**2 / (2 * R**2))
    exp_h1 = np.exp(-sigma_h1**2 * u**2 / (2 * R**2))
    exp_c  = np.exp(-sigma_c**2  * u**2 / (2 * R**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    
    term_h = (
        ((R + z_h)**2 * sin_u_Rp / up + sigma_h2**2 * cos_u_Rp) * rho_h2 * sigma_h2 * exp_h2 + 
        ((R - z_h)**2 * sin_u_Rm / um + sigma_h1**2 * cos_u_Rm) * rho_h1 * sigma_h1 * exp_h1
    )
    term_c =    (R**2 * sin_u_R  / u  + sigma_c**2  * cos_u_R)  * rho_c  * sigma_c  * exp_c
    
    res = 4 * np.pi * np.sqrt(2*np.pi) * eps * (term_h - term_c)
    return res

# ============== ellipsoid lipid bilayer shell vesicle model =====================================================
rho_c = 1


def ULV_ellipsoid(q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h, scale, c2, c3, c4):
    I_func_partial = partial(I_func, R=R, sigma_R=sigma_R, eps=eps,
                             sigma_eps=sigma_eps, rho_h=rho_h,
                             sigma_h=sigma_h, sigma_c=sigma_c, z_h=z_h)
    with Pool(processes=20) as pool:  # 6–8 is ideal on typical CPUs
        I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        #I_res = list(pool.imap(I_func_partial, q))

    return scale * np.array(I_res) + c2 * q**(-c3) + c4

def F(theta, q, R, eps, rho_h, sigma_h, sigma_c, z_h):
    u  = q * R * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    up = q * (R + z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    um = q * (R - z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    # Precompute exponentials and trig
    exp_h = np.exp(-sigma_h**2 * u**2 / (2 * R**2))
    exp_c = np.exp(-sigma_c**2 * u**2 / (2 * R**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    
    term_h = (
        (R + z_h)**2 * sin_u_Rp / up +
        (R - z_h)**2 * sin_u_Rm / um +
        sigma_h**2 * (cos_u_Rp + cos_u_Rm)
    )
    term_c = R**2 * sin_u_R / u + sigma_c**2 * cos_u_R
    
    res = (4 * np.pi * np.sqrt(2*np.pi) * eps *
           (rho_h * sigma_h * exp_h * term_h -
            rho_c * sigma_c * exp_c * term_c))
    return res

    
def P_R(r, R, sigma_R):
    """
    distribution of semi-major axis R
    """
    return 1 / (sigma_R * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((r - R) / sigma_R)**2)
    
def P_eps(p, eps, sigma_eps):
    """
    distribution of semi-minor axis eps * R
    """
    return 1 / (sigma_eps * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((p - eps) / sigma_eps)**2)

def I_func(q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h):
    #q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h = args
    
    n_R, n_eps = 50, 10
    R_vals   = np.linspace(R - 3*sigma_R, R + 3*sigma_R, n_R)
    eps_vals = np.linspace(eps - 3*sigma_eps, eps + 3*sigma_eps, n_eps)
    theta = np.linspace(0, np.pi/2, 250)  # fixed grid for integration
    sin_theta = np.sin(theta)
    
    I = 0
    for R_i in R_vals:
        for eps_j in eps_vals:
            F_vals = F(theta, q, R_i, eps_j, rho_h, sigma_h, sigma_c, z_h)
            F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)
            I += P_R(R_i, R, sigma_R) * P_eps(eps_j, eps, sigma_eps) * F_int
    
    dR = R_vals[1] - R_vals[0]
    deps = eps_vals[1] - eps_vals[0]
    return I * dR * deps

    
# ============= polydispersed homogeneous ellipsoid model =======================================================

def ellipsoid_homogen(q, R, sigma_R, eps, sigma_eps, scale, c4):
    """
    Computes intensity I(q) for a polydispersed homogeneous ellipsoid.
    Parallelized over q using multiprocessing Pool.
    """
    with Pool(processes=6) as pool:  # 6–8 is usually efficient
        I_res = list(tqdm(pool.imap(partial(integrate_F_ellipsoid_homogen, R=R, sigma_R=sigma_R, eps=eps, sigma_eps=sigma_eps), q),
                          total=len(q), desc="Computing I(q)", ncols=80))
    return scale * np.array(I_res) + c4
                       
# def ellipsoid_homogen(q, R, sigma_R, eps, sigma_eps, scale):
#     # Prepare arguments for parallel map
#     args_list = [(q_val, R, sigma_R, eps, sigma_eps) for q_val in q]
    
#     with Pool(processes=10) as pool:
#         I_res = list(tqdm(pool.imap(integrate_F_ellipsoid_homogen, args_list),
#                           total=len(args_list),
#                           desc="Computing I(q)",
#                           ncols=80))
    
#     # threshold = 1e-8
#     # for i in range(len(I_res)):
#     #     if I_res[i] < threshold:
#     #         I_res[i] = (I_res[i+1] + I_res[i-1])/2
    
#     return scale * np.array(I_res)

def F_ellipsoid_homogen(theta, q, R, eps):
    """
    Vectorized form factor of a homogeneous ellipsoid.
    theta can be an array.
    """
    u = q * R * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    volume = 4/3 * np.pi * eps * R**3
    return 3 * volume * (np.sin(u) - u * np.cos(u)) / (u + 1e-15)**3
    
# def F_ellipsoid_homogen(q, R, eps, theta):
#     u      = q * R * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
#     volume = 4/3 * np.pi * eps * R**3
#     res    = 3 * volume * (np.sin(u) - u * np.cos(u)) / u**3 
#     return res


# def integrate_F_ellipsoid_homogen(args):
#     q, R, sigma_R, eps, sigma_eps = args
    
#     delta_R   = 0.5 * sigma_R
#     delta_eps = 0.5 * sigma_eps
    
#     start_R   = R   - 3 * sigma_R
#     start_eps = eps - 3 * sigma_eps
    
#     step_num = 12
#     a = 0
#     b = np.pi / 2
#     I = 0
#     for i in range(step_num):
#         for j in range(step_num):
#             R_i   = (i + 0.5) * delta_R
#             eps_j = (j + 0.5) * delta_eps
#             def kernel_ellips_hom(theta):
#                 return np.abs(F_ellipsoid_homogen(q, R, eps, theta))**2 * np.sin(theta)
            
#             F_integrated = scipy.integrate.quad(kernel_ellips_hom, a, b, limit=300)[0]
#             I = I + delta_R * delta_eps * P_R(R_i, R, sigma_R) * P_eps(eps_j, eps, sigma_eps) * F_integrated
#     return I

def integrate_F_ellipsoid_homogen(q, R, sigma_R, eps, sigma_eps,
                                  n_R=12, n_eps=12, n_theta=150):
    """
    Compute I(q) for one q value, averaged over distributions in R and eps.
    Vectorized over theta and uses np.trapezoid for fast integration.
    """
    R_vals   = np.linspace(R - 3*sigma_R, R + 3*sigma_R, n_R)
    eps_vals = np.linspace(eps - 3*sigma_eps, eps + 3*sigma_eps, n_eps)
    theta = np.linspace(0, np.pi/2, n_theta)
    sin_theta = np.sin(theta)

    # Precompute probability weights
    P_R_vals = P_R(R_vals, R, sigma_R)
    P_eps_vals = P_eps(eps_vals, eps, sigma_eps)

    I = 0.0
    for i, R_i in enumerate(R_vals):
        for j, eps_j in enumerate(eps_vals):
            F_vals = F_ellipsoid_homogen(theta, q, R_i, eps_j)
            F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)
            I += P_R_vals[i] * P_eps_vals[j] * F_int

    dR = R_vals[1] - R_vals[0]
    deps = eps_vals[1] - eps_vals[0]
    return I * dR * deps


#=============== monodispersed hard sphere model =====================================================================

def hard_sphere_raw(q, r):
    F = 3 * (np.sin(q * r) - q * r * np.cos(q * r)) / (q**3 * r**3)  
    return F

def hard_sphere(q, R, scale, c2, c3, c4):
    V = 4/3 * np.pi * R**3
    return scale * V**2 * np.abs(hard_sphere_raw(q, R))**2 + c2 * q**(-c3) + c4


#=============== POLYDISPERSED hard sphere model ========================================================================

def hard_sphere_polydisp_NO_BGR(q, R, sigma_R, scale):

    # Prepare arguments for parallel map
    args_list = [(q_val, R, sigma_R) for q_val in q]
    
    with Pool(processes=10) as pool:  # Adjust processes to your server's cores
        I_res = np.array(pool.map(integrate_for_q, args_list))
    # threshold = 1e-8
    # for i in range(len(I_res)):
    #     if I_res[i] < threshold:
    #         I_res[i] = (I_res[i+1] + I_res[i-1])/2
    return scale * I_res 

def integrate_for_q(args, n_pts=1000):
    q_val, R, sigma_R = args

    # Integration range: cover +5σ around mean radius
    r_min = max(0, R - 5 * sigma_R)
    r_max = R + 5 * sigma_R

    # Discretize radius axis
    r = np.linspace(r_min, r_max, n_pts)

    # Gaussian distribution
    gaussian = (1 / (sigma_R * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - R) / sigma_R)**2)

    # Volume and intensity kernel
    volume = (4/3) * np.pi * r**3

    kernel = gaussian * volume**2 * np.abs(hard_sphere_raw(q_val, r))**2

    # Integrate using the trapezoid rule
    I_integrated = np.trapezoid(kernel, r)

    return I_integrated
    
def hard_sphere_polydisp(q, R, sigma_R, scale, c2, c3, c4):

    
    # Prepare arguments for parallel map
    args_list = [(q_val, R, sigma_R) for q_val in q]
    
    with Pool(processes=10) as pool:  # Adjust processes to your server's cores
        I_res = np.array(pool.map(integrate_for_q, args_list))
    
    return scale * I_res + c4 + c2 * q**(-c3) 

#========================== symmetric flat bilayer model NO BGR ==========================

# eq. 6 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)
# with symmetric rho, sigma, epsilon

def ULV_flat_sym_NO_BGR(q, sigma_head, sigma_tail, rho_head, eps_head, scale):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([rho_head, -1, rho_head])
    epsilon = np.array([-eps_head, 0, eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result 

#========================== symmetric flat bilayer model CONST BGR ==========================

# eq. 6 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)
# with symmetric rho, sigma, epsilon

def ULV_flat_sym_CONST_BGR(q, sigma_head, sigma_tail, rho_head, eps_head, scale, c4):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([rho_head, -1, rho_head])
    epsilon = np.array([-eps_head, 0, eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result + c4

#========================== symmetric flat bilayer model  ==========================

# eq. 6 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)
# with symmetric rho, sigma, epsilon

def ULV_flat_sym(q, sigma_head, sigma_tail, rho_head, eps_head, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([rho_head, -1, rho_head])
    epsilon = np.array([-eps_head, 0, eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result + c2 * q**(-c3) + c4

# ========================== asymmetric flat bilayer model  ==========================

# eq. 6 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)
# with asymmetric rho, sigma

def ULV_flat_asym(q, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_head, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_h1, sigma_tail,  sigma_h2])
    rho     = np.array([rho_h1,           -1,    rho_h2])
    epsilon = np.array([-eps_head,         0,  eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result + c2 * q**(-c3) + c4


# ========================== NO POWER LAW asymmetric flat bilayer model  ==========================

# eq. 6 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)
# with asymmetric rho, sigma

def ULV_flat_asym_no_powlaw(q, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_head, scale, c4):
    
    sigma   = np.array([sigma_h1, sigma_tail, sigma_h2])
    rho     = np.array([rho_h1, -1, rho_h2])
    epsilon = np.array([-eps_head, 0, eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result + c4

# ========================== symmetric Brunger model fit function without sigma_R ==========================

# eq. 7 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowiczb and Brunger (2005)

def ULV_without_sigma_R(q, R, sigma_head, sigma_tail, rho_head, eps_head, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([rho_head, -1, rho_head])
    epsilon = np.array([-eps_head, 0, eps_head])
    
    result  = 0
    for i in range(3):
        for j in range(3):
            result += (R + epsilon[i]) * (R + epsilon[j]) * rho[i]*rho[j]*sigma[i]*sigma[j] * np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2)/2) * np.cos(q * (epsilon[i] - epsilon[j]))
    
    return scale/q**2 * result + c2 * q**(-c3) + c4



# ========================== symmetric full Brunger model fit function ==========================

# eq. 12 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowicz and Brunger (2005)


def ULV_spheric_sym(q, R, sigma_R, sigma_head, sigma_tail, rho_head, eps_head, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([  rho_head,         -1,   rho_head])
    eps     = np.array([ -eps_head,          0,   eps_head])

    result = 0
    for i in range(3):  
        for j in range(3):
            result += ( rho[i]*rho[j]*sigma[i]*sigma[j] *
                        np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2) / 2) * 
                        (A(q, R, sigma_R, eps[i], eps[j]) - B(q, R, sigma_R, eps[i], eps[j]) + 
                         C(q, R, sigma_R, eps[i], eps[j])) )

    return scale/q**2 * result + c2 * q**(-c3) + c4


# ========================== REDUCED symmetric full Brunger model fit function ==========================

# eq. 12 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowicz and Brunger (2005)
# Tim's tip
# is it useful? --> see fit_ULV_without_sigma_R


def ULV_reduced_spheric_sym(q, R, sigma_R, sigma_head, sigma_tail, rho_head, eps_head, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_head, sigma_tail, sigma_head])
    rho     = np.array([  rho_head,         -1,   rho_head])
    eps     = np.array([ -eps_head,          0,   eps_head])

    result = 0
    for i in range(3):  
        for j in range(3):
            result += ( rho[i]*rho[j]*sigma[i]*sigma[j] *
                        np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2) / 2) * 
                        A(q, R, sigma_R, eps[i], eps[j]) )
                      

    return scale/q**2 * result + c2 * q**(-c3) + c4


# ========================== REDUCED ASYMMETRIC full Brunger model fit function ==========================

# eq. 12 of "X-ray scattering from unilamellar lipid vesicles" by Brzustowicz and Brunger (2005)
# Tim's tip
# asymmetric
# is it useful? --> see fit_ULV_without_sigma_R


def ULV_reduced_spheric_asym(q, R, sigma_R, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_h1, eps_h2, scale, c2, c3, c4):
    
    sigma   = np.array([sigma_h1, sigma_tail, sigma_h2])
    rho     = np.array([  rho_h1,         -1,   rho_h2])
    eps     = np.array([ -eps_h1,          0,   eps_h2])

    result = 0
    for i in range(3):  
        for j in range(3):
            result += ( rho[i]*rho[j]*sigma[i]*sigma[j] *
                        np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2) / 2) * 
                        A(q, R, sigma_R, eps[i], eps[j]) )
                      

    return scale/q**2 * result + c2 * q**(-c3) + c4

# ========================== FULLY ASYMMETRIC full Brunger model fit function ==========================

# that means sigma and rho are asymmetric

def A(q, R, sigma_R, eps_i, eps_j):

    return ((R + eps_i) * (R + eps_j) + sigma_R**2) * np.cos(q * (eps_i - eps_j))

def B(q, R, sigma_R, eps_i, eps_j):

    return np.exp(-2 * q**2 * sigma_R**2) * ((R + eps_i) * (R + eps_j) + sigma_R**2 - 4*q**2 * sigma_R**4) * np.cos(q * (2*R + eps_i + eps_j))

def C(q, R, sigma_R, eps_i, eps_j):

    return q * sigma_R**2 * np.exp(-2 * q**2 * sigma_R**2) * (2*R + eps_i + eps_j) * np.sin(q * (2*R + eps_i + eps_j))
                    
def ULV_spheric_asym(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c2, c3, c4):
    
    sigma   = np.array([  sigma_h1,    sigma_c,   sigma_h2])
    rho     = np.array([    rho_h1,         -1,     rho_h2])
    eps     = np.array([      -z_h,          0,        z_h])

    result = 0
    for i in range(3):  
        for j in range(3):
            result += ( rho[i]*rho[j]*sigma[i]*sigma[j] *
                        np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2) / 2) * 
                        (A(q, R, sigma_R, eps[i], eps[j]) - B(q, R, sigma_R, eps[i], eps[j]) + 
                        C(q, R, sigma_R, eps[i], eps[j])) )

    return scale/q**2 * result + c2 * q**(-c3) + c4

# ========================== NO BACKGROUND fully asymmetric full Brunger model fit function ==========================

# that means epsilon, sigma and rho are asymmetric

def ULV_spheric_asym_NO_BGR(q, R, sigma_R, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_head, scale):
    
    sigma   = np.array([sigma_h1,   sigma_tail,   sigma_h2])
    rho     = np.array([  rho_h1,           -1,     rho_h2])
    eps     = np.array([ -eps_head,          0,   eps_head])

    result = 0
    for i in range(3):  
        for j in range(3):
            result += ( rho[i]*rho[j]*sigma[i]*sigma[j] *
                        np.exp(-q**2 * (sigma[i]**2 + sigma[j]**2) / 2) * 
                        (A(q, R, sigma_R, eps[i], eps[j])- B(q, R, sigma_R, eps[i], eps[j]) + 
                        C(q, R, sigma_R, eps[i], eps[j])) )

    return scale/q**2 * result 

#=========== spherical ULV model with numerical integration over polydispersity =====================================

# not exactly like the "analytical" BB model 

def ULV_spheric_asym_num(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c2, c3, c4):
    I_func_partial = partial(spheric_asym_integrator, R=R, sigma_R=sigma_R, rho_h1=rho_h1, rho_h2=rho_h2,
                             sigma_h1=sigma_h1, sigma_h2=sigma_h2, sigma_c=sigma_c, z_h=z_h)
    with Pool(processes=9) as pool: 
        # I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        I_res = pool.map(I_func_partial, q)

    return scale * np.array(I_res) + c2 * q**(-c3) + c4    

def ULV_spheric_sym_num(q, R,    sigma_R, sigma_h, sigma_c, rho_h, z_h,    scale,      c2,      c3,      c4):
    I_func_partial = partial(spheric_asym_integrator, R=R, sigma_R=sigma_R, rho_h1=rho_h, rho_h2=rho_h,
                             sigma_h1=sigma_h, sigma_h2=sigma_h, sigma_c=sigma_c, z_h=z_h)
    with Pool(processes=9) as pool: 
        # I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        I_res = pool.map(I_func_partial, q)

    return scale * np.array(I_res) + c2 * q**(-c3) + c4   

def spheric_asym_integrator(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h):
    #q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h = args
    
    n_R    = 100
    R_vals = np.linspace(R - 4*sigma_R, R + 4*sigma_R, n_R)    
    I      = np.sum(P_R(R_vals, R, sigma_R) * np.abs(F_spheric_asym(q, R_vals, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h))**2)
    dR     = R_vals[1] - R_vals[0]
    return I * dR 

def F_spheric_asym(q, R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h):
    u  = q *  R        
    up = q * (R + z_h) 
    um = q * (R - z_h) 
    # Precompute exponentials and trig
    exp_h2 = np.exp(-sigma_h2**2 * u**2 / (2 * R**2))
    exp_h1 = np.exp(-sigma_h1**2 * u**2 / (2 * R**2))
    exp_c  = np.exp(-sigma_c**2  * u**2 / (2 * R**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    
    term_h = (
        ((R + z_h)**2 * sin_u_Rp / up + sigma_h2**2 * cos_u_Rp) * rho_h2 * sigma_h2 * exp_h2 + 
        ((R - z_h)**2 * sin_u_Rm / um + sigma_h1**2 * cos_u_Rm) * rho_h1 * sigma_h1 * exp_h1
    )
    term_c =    (R**2 * sin_u_R  / u  + sigma_c**2  * cos_u_R)  * rho_c  * sigma_c  * exp_c
    
    res = 4 * np.pi * np.sqrt(2*np.pi) * (term_h - term_c)
    return res

#=========== spherical ULV model with numerical integration over LOG NORMAL polydispersity =====================================
# ============= Log-normal distribution for polydispersity ==========================================
from scipy.stats import lognorm
def lognormal_polydisp(R, sigma, R0):
    P = lognorm.pdf(R, s=sigma, scale=R0)
    return P
# R0 is the median of the distribution, sigma is the shape parameter (not standard deviation) !!!

def ULV_spheric_asym_lognormal(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, scale, c2, c3, c4):
    I_func_partial = partial(spheric_asym_lognormal_integrator, R=R, sigma_R=sigma_R, rho_h1=rho_h1, rho_h2=rho_h2,
                             sigma_h1=sigma_h1, sigma_h2=sigma_h2, sigma_c=sigma_c, z_h=z_h)
    with Pool(processes=9) as pool: 
        I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        # I_res = pool.map(I_func_partial, q)

    return scale * np.array(I_res) + c2 * q**(-c3) + c4      

def spheric_asym_lognormal_integrator(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h):
    #q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h = args
    n_R    = 1000
    R_vals = np.linspace(1e-1, 70, n_R)    
    I      = np.trapezoid(lognormal_polydisp(R_vals, sigma_R, R) * np.abs(F_spheric_asym(q, R_vals, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h))**2, R_vals)
    return I 

# ================== model for ellipsoid ULV (asym. EDP) with additional outer protein layer =============================

def ULV_ell_asym_prot_layer(q, R, sigma_R, eps, sigma_eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c2, c3, c4):
    I_func_partial = partial(ell_asym_prot_integrator, R=R, sigma_R=sigma_R, eps=eps,
                             sigma_eps=sigma_eps, rho_h1=rho_h1, rho_h2=rho_h2,
                             sigma_h1=sigma_h1, sigma_h2=sigma_h2, sigma_c=sigma_c, z_h=z_h, z_h3=z_h3, sigma_h3=sigma_h3, rho_h3=rho_h3)
    with Pool(processes=12) as pool:  # 6–8 is ideal on typical CPUs
        I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        #I_res = pool.map(I_func_partial, q)
    
    return scale * np.array(I_res) + c2 * q**(-c3) + c4

def ell_asym_prot_integrator(q, R, sigma_R, eps, sigma_eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3):
    #q, R, sigma_R, eps, sigma_eps, rho_h, sigma_h, sigma_c, z_h = args
    
    n_R, n_eps = 60, 10
    R_vals   = np.linspace(R   - 3*sigma_R,     R + 3*sigma_R,     n_R)
    eps_vals = np.linspace(eps - 3*sigma_eps, eps + 3*sigma_eps, n_eps)
    theta = np.linspace(0, np.pi/2, 250)  # fixed grid for integration
    sin_theta = np.sin(theta)
    
    I = 0
    for R_i in R_vals:
        for eps_j in eps_vals:
            F_vals = F_ell_asym_prot(theta, q, R_i, eps_j, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3)
            F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)         
            I += P_R(R_i, R, sigma_R) * F_int * P_eps(eps_j, eps, sigma_eps)
    
    dR = R_vals[1] - R_vals[0]
    deps = eps_vals[1] - eps_vals[0]
    return I * dR * deps

def F_ell_asym_prot(theta, q, R, eps, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3):
    u     = q *  R        * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    up    = q * (R + z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    um    = q * (R - z_h) * np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)
    uprot = q * (R + z_h3)* np.sqrt(np.sin(theta)**2 + eps**2 * np.cos(theta)**2)

    exp_h2 = np.exp(-sigma_h2**2 * u**2 / (2 * R**2))
    exp_h1 = np.exp(-sigma_h1**2 * u**2 / (2 * R**2))
    exp_c  = np.exp(-sigma_c**2  * u**2 / (2 * R**2))
    exp_h3 = np.exp(-sigma_h3**2  * u**2 / (2 * R**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    sin_u_prot = np.sin(uprot)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    cos_u_prot = np.cos(uprot)
    
    term_h = (
        ((R + z_h)**2 * sin_u_Rp / up + sigma_h2**2 * cos_u_Rp) * rho_h2 * sigma_h2 * exp_h2 + 
        ((R - z_h)**2 * sin_u_Rm / um + sigma_h1**2 * cos_u_Rm) * rho_h1 * sigma_h1 * exp_h1
    )
    term_c =    (R**2 * sin_u_R  / u  + sigma_c**2  * cos_u_R)  * rho_c  * sigma_c  * exp_c
    term_prot = ((R + z_h3)**2 * sin_u_prot / uprot + sigma_h3**2 * cos_u_prot) * rho_h3 * sigma_h3 * exp_h3
    
    res = 4 * np.pi * np.sqrt(2*np.pi) * eps * (term_h - term_c + term_prot)
    return res

# ================== model for spherical ULV (asym. EDP) with additional outer protein layer =============================

def ULV_sph_asym_prot_layer(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3, scale, c2, c3, c4):
    I_func_partial = partial(sph_asym_prot_integrator, R=R, sigma_R=sigma_R, rho_h1=rho_h1, rho_h2=rho_h2,
                             sigma_h1=sigma_h1, sigma_h2=sigma_h2, sigma_c=sigma_c, z_h=z_h, z_h3=z_h3, sigma_h3=sigma_h3, rho_h3=rho_h3)
    with Pool(processes=12) as pool:  # 6–8 is ideal on typical CPUs
        I_res = list(tqdm(pool.imap(I_func_partial, q), total=len(q), desc="Computing I(q)", ncols=80))
        #I_res = pool.map(I_func_partial, q)
    
    return scale * np.array(I_res) + c2 * q**(-c3) + c4

def sph_asym_prot_integrator(q, R, sigma_R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3):
    
    n_R = 100
    R_vals   = np.linspace(R   - 3*sigma_R,     R + 3*sigma_R,     n_R)
    theta = np.linspace(0, np.pi/2, 250)  # fixed grid for integration
    sin_theta = np.sin(theta)
    
    I = 0
    for R_i in R_vals:
        F_vals = F_sph_asym_prot(theta, q, R_i, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3)
        F_int = np.trapezoid(np.abs(F_vals)**2 * sin_theta, theta)         
        I += P_R(R_i, R, sigma_R) * F_int 
    
    dR = R_vals[1] - R_vals[0]
    return I * dR 

def F_sph_asym_prot(theta, q, R, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, z_h3, sigma_h3, rho_h3):
    u     = q *  R        
    up    = q * (R + z_h) 
    um    = q * (R - z_h) 
    uprot = q * (R + z_h3)

    exp_h2 = np.exp(-sigma_h2**2 * u**2 / (2 * R**2))
    exp_h1 = np.exp(-sigma_h1**2 * u**2 / (2 * R**2))
    exp_c  = np.exp(-sigma_c**2  * u**2 / (2 * R**2))
    exp_h3 = np.exp(-sigma_h3**2  * u**2 / (2 * R**2))
    
    sin_u_Rp = np.sin(up)
    sin_u_Rm = np.sin(um)
    sin_u_R  = np.sin(u)
    sin_u_prot = np.sin(uprot)
    
    cos_u_Rp = np.cos(up)
    cos_u_Rm = np.cos(um)
    cos_u_R  = np.cos(u)
    cos_u_prot = np.cos(uprot)
    
    term_h = (
        ((R + z_h)**2 * sin_u_Rp / up + sigma_h2**2 * cos_u_Rp) * rho_h2 * sigma_h2 * exp_h2 + 
        ((R - z_h)**2 * sin_u_Rm / um + sigma_h1**2 * cos_u_Rm) * rho_h1 * sigma_h1 * exp_h1
    )
    term_c =    (R**2 * sin_u_R  / u  + sigma_c**2  * cos_u_R)  * rho_c  * sigma_c  * exp_c
    term_prot = ((R + z_h3)**2 * sin_u_prot / uprot + sigma_h3**2 * cos_u_prot) * rho_h3 * sigma_h3 * exp_h3
    
    res = 4 * np.pi * np.sqrt(2*np.pi) * (term_h - term_c + term_prot)
    return res

# ========================== ## Electron Density Profile (EDP) plot function ==========================
def q_minus_4(q, f):
    return q**(-4) * f
    
def gaussian(r, rho_pos, sigma, rho_h):

    return rho_h * np.exp(-0.5 * ((r - rho_pos) / sigma)**2)

def EDP_analysis(params):
    R = 0
    rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h = params
    r = np.linspace(R - 2*z_h, R + 2*z_h, 10000)
    rho_h   = [rho_h1, -1, rho_h2]
    rho_pos = [R - z_h, R, R + z_h]
    sigma   = [sigma_h1, sigma_c, sigma_h2]
    
    rho = np.zeros_like(r)
    for i in range(3):
        rho += gaussian(r, rho_pos[i], sigma[i], rho_h[i])
    root_indices = []
    for i in range(len(r)):
        if ((rho[i-1]>0 and rho[i]<0) or (rho[i-1]<0 and rho[i]>0)) and (np.abs(r[i]) < z_h):
            root_indices.append(i)
    rho_1 = np.trapezoid(rho[:root_indices[0]], r[:root_indices[0]])
    rho_2 = np.trapezoid(rho[root_indices[0]:root_indices[1]], r[root_indices[0]:root_indices[1]])
    rho_3 = np.trapezoid(rho[root_indices[1]:], r[root_indices[1]:])
    
    EDP_tot     = rho_1 + rho_2 + rho_3
    EDP_hg      = rho_1 + rho_3
    EDP_tg      = np.abs(rho_2)
    EDP_hg_asym = np.abs(rho_1 - rho_3) / EDP_hg
    EDP_hg_tg   = np.abs(EDP_hg / rho_2)
    
    return EDP_tot, EDP_hg, EDP_tg, EDP_hg_asym, EDP_hg_tg
    
def EDP_plot(R, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_h1, eps_h2, info_text, title, z_h3=2.2, sigma_h3=1, rho_h3=0):

    R = 0
    if rho_h3 != 0:
        r = np.linspace(R - 2*eps_h1, R + 6*eps_h2, 1500)
    else: 
        r = np.linspace(R - 2*eps_h1, R + 3*eps_h2, 1500)
    rho_h   = [rho_h1, -1, rho_h2, rho_h3]
    rho_pos = [R - eps_h1, R, R + eps_h2, R + z_h3]
    sigma   = [sigma_h1, sigma_tail, sigma_h2, sigma_h3]

    rho = np.zeros_like(r)
    for i in range(4):
        rho += gaussian(r, rho_pos[i], sigma[i], rho_h[i])
    EDP_int = np.trapezoid(rho, r) 
    
    fig = plt.figure(figsize=(12, 7.5))
    plt.plot(r, rho, color='darkgreen', label='EDP')
    plt.xlabel('$r[\\mathrm{nm}]$')
    plt.ylabel('$\\rho\\mathrm{[a.u.]}$')
    plt.title(title)
    plt.grid(True)
    # xmax = plt.xlim()[1]
    # ypos = 1 # middle of y-range
    # plt.text(xmax * 1.01, ypos, info_text, va='center', fontsize=12, color='black')
    legend = plt.legend()
    EDP_area = f"EDP_area = {EDP_int:.3g}" 
    legend.set_title(EDP_area)
    return fig

def EDP_plot_together(param_list, title):
    """
    Plot several EDPs together.
    Parameters:
        param_list : list of lists or tuples                 
            Each element = [sample, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, color]
    """
    R = 0
    fig = plt.figure(figsize=(12, 7.5))

    for params in param_list:
        sample, rho_h1, rho_h2, sigma_h1, sigma_h2, sigma_c, z_h, color= params
        r = np.linspace(R - 2*z_h, R + 2*z_h, 10000)
        rho_h   = [rho_h1, -1, rho_h2]
        rho_pos = [R - z_h, R, R + z_h]
        sigma   = [sigma_h1, sigma_c, sigma_h2]
        rho = np.zeros_like(r)
        for i in range(3):
            rho += gaussian(r, rho_pos[i], sigma[i], rho_h[i])
        # analysis = EDP_analysis(params[:-1])
        if sample.startswith("flat asym."):
            plt.plot(r, rho, label=sample, color=color, ls="--")
        else:
            plt.plot(r, rho, label=sample, color=color, lw=3)
    
    plt.xlabel('$z[\\mathrm{nm}]$', fontsize=24)
    plt.ylabel('$\\rho[\\mathrm{a.u.}]$', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.title(title)
    plt.grid(True)
    # plt.legend()
    # legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
    # legend.set_title(info_text)
    return fig

#============= EDP data ======================================================

EDP_lib = {
    # Each value =          [sample,                    rho_h1, rho_h2, sigma_h1, sigma_h2,  sigma_c,    z_h]
    "HA5_CA5_LV4_25uMaSyn": ["HA5_CA5_LV4_25uMaSyn",    1.4986, 1.7302,  0.37497,  0.20002,  0.75887, 1.8633],
    "HA5_CA4_LV4_15uMaSyn": ["HA5_CA4_LV4_15uMaSyn",    1.5328, 1.6601,  0.35989,  0.20032,  0.71574, 1.8356],
    "HA5_CA6_LV4_9uMaSyn":  ["HA5_CA6_LV4_9uMaSyn",     1.3512, 1.8002,  0.37945,  0.20025,  0.74247, 1.8666],
    
    "HA3_b_CA4_LV4_7p5aSyn":["HA3_b_CA4_LV4_7p5aSyn",  1.478,  1.7475,  0.36964,  0.20011,  0.73867, 1.7677],
    "HA3_b_CA5_LV4_3aSyn":  ["HA3_b_CA5_LV4_3aSyn",     1.4598, 1.7853,  0.34637,  0.20003,  0.7235,  1.8187], 
    "HA3_b_CA6_LV4_1aSyn":  ["HA3_b_CA6_LV4_1aSyn",     1.2544, 1.5019,  0.35065,  0.20000,  0.608,   1.8441], 
    "HA3_b_CA7_LV4_0p3aSyn":["HA3_b_CA7_LV4_0p3aSyn",   1.2739, 1.7445,  0.36741,  0.20000,  0.68632, 1.8015], 
    # "HA3_b_CA8_LV4_0p1aSyn": [],   FIT FAILED
    
    "HA3_CA9_LV4a": {
        "flat_sym_reduced_q":    ["flat sym. (high-q)", 1.2256, 1.2256,  0.3929,   0.3929,   0.92876, 1.7992, "springgreen"], # q > 0.7 1/nm
        "flat_sym_all_q":            ["flat sym.", 1.1562, 1.1562,  0.39919,  0.39919,  0.80971, 1.8422, "green"],

        "flat_asym_reduced_q": ["flat asym. (high-q)",  1.1118, 1.0721,  0.30577,  0.31556,  0.64926, 1.8948, "cyan"], # q > 0.7 1/nm
        "flat_asym_all_q":          ["flat asym.", 0.9077, 1.6139,  0.472,    0.30579,  0.8079,  1.8439, "tomato"],

        "spheric_asym_restricted_right_max":    ["spherical asym.",    1.015,  1.895,   0.508,    0.2,      0.7631, 1.857, "darkorange"], 
        # op.error, with weight, rho in [1, 4], sigma in [0.2, 0.6] 
        "spheric_asym_restricted_left_max": ["",        1.329,  1,       0.3798,   0.3688,   0.7441, 1.861, "firebrick"], 
        # op.error, with weight, rho in [1, 4], sigma in [0.2, 0.6]
        "spheric_asym_free_right_max": ["",             0.9519, 4,       0.531,    0.0934,   0.7487, 1.863, "deepskyblue"], 
        # op.error, with weight, rho in [0, 4], sigma in [0, 0.6]
        "spheric_asym_free_left_max": ["",              4,      0.704,   0.1259,   0.5158,   0.7356, 1.868, "green"], 
        # op.error, with weight, rho in [0, 4], sigma in [0, 0.6]
        
        "ellipsoid_asym":               ["ellipsoid asym.", 1.8799, 1.0389,  0.30377,  0.44115,  0.89864, 1.7498, "red"],
    }
}
# ==============================================================================================
best_fits = {
    # =========================== LV4 + aSyn ==============================================================
    "HA3_CA9_LV4a": {
        # "flat_sym_all_q":     [0.39919, 0.80971, 1.1562, 1.8422, 0.49838,    0.29, 0.0080396, -0.27696],  
        # op.error, with weight 
        "flat_sym_reduced_q": [0.3929,  0.92876, 1.2256, 1.7992,     0.5, 0.28428,     1e-10, -0.27424],   
        # q > 0.7,  chi2 = 2.5921, error not operative!!
        "flat_asym_reduced_q": [0.30577,  0.31556, 0.64926, 1.1118, 1.0721, 1.8948, 0.60182, 0.23937, 0.0048118, -0.22762], 
        # q > 0.7, chi2 = 2.1709, error not operative!!
        "flat_asym_all_q":     [0.472,    0.30579, 0.8079,  0.9077, 1.6139, 1.8439, 0.49691, 0.28398, 1.8061e-11, -0.27496], 
        # op. error
        "ellipsoid_asym": [9.9997, 5.9694, 1.1317, 0.10304, 1.8799, 1.0389, 0.30377, 0.44115, 0.89864, 1.7498, 6.2771e-06, 0, 0, 0.0086], 
        # operative error, with weight for q < 0.1, chi2 = 
        
        # =======================================================
        "spheric_asym_restricted_right_max": [9.515, 5.395, 1.015, 1.895, 0.508, 0.2, 0.7631, 1.857, 0.004288, 3.895, 3.811e-14, -3.887], 
        # op.error, with weight, rho in [1, 4], sigma in [0.2, 0.6]
        "spheric_asym_restricted_left_max": [11.49, 4.622, 1.329, 1, 0.3798, 0.3688, 0.7441, 1.861, 0.003367, 3.941, 0.0003279, -3.929], 
        # op.error, with weight, rho in [1, 4], sigma in [0.2, 0.6]
        "spheric_asym_free_right_max": [9.008, 5.51, 0.9519, 4, 0.531, 0.0934, 0.7487, 1.863, 0.004641, 3.934, 1.656e-38, -3.928], 
        # op.error, with weight, rho in [0, 4], sigma in [0, 0.6]
        "spheric_asym_free_left_max": [9.215, 5.448, 4, 0.704, 0.1259, 0.5158, 0.7356, 1.868, 0.004542, 3.951, 1.375e-39, -3.944], 
        # op.error, with weight, rho in [0, 4], sigma in [0, 0.6]

        # =========================================================
        # Feb16
        1: [9.4014, 5.4172, 0.99871, 2.1, 0.51263, 0.17876, 0.75606, 1.8594, 0.0043783, 0, 0, 0.0073947],
        2: [9.0076, 5.5102, 0.95189, 4, 0.53097, 0.093403, 0.74874, 1.8634, 0.004641, 0, 0, 0.005863],
        3: [10.418, 5.2498, 1.2, 1.4807, 0.51575, 0.30752, 0.93462, 1.8101, 0.0034883, 0, 0, 0.0087973],
        4: [11.126, 5.1881, 1.7927, 1, 0.36648, 0.48452, 1, 1.8022, 0.0030718, 0, 0, 0.0094042],
        5: [10.057, 5.1207, 1.3376, 1, 0.37501, 0.36243, 0.73382, 1.8639, 0.0041109, 0, 0, 0.010015],
        6: [9.6442, 5.3449, 2.5, 0.74074, 0.20402, 0.49463, 0.7445, 1.8641, 0.0042505, 0, 0, 0.0078341],
        7: [9.2148, 5.4482, 4, 0.70397, 0.12588, 0.51581, 0.73561, 1.8685, 0.0045418, 0, 0, 0.0064349],
        # 8: [25, 8, 1, 1.501, 0.47575, 0.31641, 0.83581, 1.8355, 0.00071119, 0, 0, 0.0090776],
        9: [10.117, 5.0551, 1.1327, 2.3532, 0.5883, 0.2, 1, 1.7968, 0.0037706, 0, 0, 0.0046466],
        10:[9.3549, 4.9981, 1.0814, 2.4607, 0.57286, 0.2, 0.98006, 1.8046, 0.0041635, 0, 0, 0.012664],
        11:[11.232, 5.0942, 1.0399, 1.851, 0.51616, 0.2, 0.77481, 1.8519, 0.0033537, 0, 0, 0.01118],
    },
    # ==================================
    "HA5_CA5_LV4_25uMaSyn": {
        "ellipsoid_asym": [10.471, 7.133, 0.98558, 0.21418, 1.4986, 1.7302,  0.37497,  0.20002,  0.75887, 1.8633, 8.08e-6, 3.025e-6, 0.017753, 0.010487],
        "spheric_asym": {
            1: [8.3893, 5.825, 1.0351, 2.5, 0.57397, 0.21676, 1, 1.803, 0.0055907, 0, 0, 0.0096369],
        }
    },
    "HA5_CA4_LV4_15uMaSyn": {
        "ellipsoid_asym": [10.847, 6.0374, 1.0642, 0.24972, 1.5328, 1.6601,  0.35989,  0.20032,  0.71574, 1.8356, 7.303e-6, 2.9847e-5, 0.0061981, 0.010649],
        "spheric_asym": {
            1: [6.9571, 5.6057, 1.1159, 2.1, 0.56294, 0.24052, 0.97244, 1.8054, 0.0067085, 0, 0, 0.0059828],
            2: [6.5018, 6.0877, 1.0733, 2.7, 0.58505, 0.17811, 0.94866, 1.8074, 0.0068877, 0, 0, 0.0042884],
            3: [6.3044, 6.3366, 1.0398, 4, 0.6, 0.11812, 0.93745, 1.8088, 0.0069044, 0, 0, 0.0025293],
            4: [6.5386, 6.104, 4, 0.82094, 0.15982, 0.6, 0.97236, 1.8018, 0.0068808, 0, 0, 0.0036727],
            5: [6.5873, 7.6356, 2.872, 0.9, 0.22278, 0.57385, 1, 1.7998, 0.0053146, 0, 0, 0.0056559],
            6: [6.5366, 7.9999, 2.2876, 1, 0.27991, 0.51628, 1, 1.802, 0.0049525, 0, 0, 0.0076825],
            7: [12.304, 4.6158, 2.6174, 0.77582, 0.26596, 0.53002, 0.92634, 1.8162, 0.0030657, 0, 0, 0.006648],
            8: [14.02, 4.544, 2.1, 0.81908, 0.32897, 0.51483, 0.93725, 1.8084, 0.0025269, 0, 0, 0.0085101],
            9: [8.1206, 4.924, 1.0538, 2.5, 0.57752, 0.21841, 1, 1.8013, 0.005889, 0, 0, 0.010096],
        }
    },
    "HA5_CA6_LV4_9uMaSyn": {
        "ellipsoid_asym": [10.871, 7.0448, 0.99143, 0.21195, 1.3512, 1.8002,  0.37945,  0.20025,  0.74247, 1.8666, 8.026e-6, 2.69e-5, 0.05624, 0.012356],
        "spheric_asym": {
            1: [10.29, 6.0461, 0.99299, 2.1, 0.55385, 0.22156, 0.8889, 1.8221, 0.0043911, 0, 0, 0.0084407],
            2: [9.9523, 6.076, 0.89834, 4, 0.56279, 0.1094, 0.82042, 1.8429, 0.0047074, 0, 0, 0.0054722],
            3: [9.5077, 5.6095, 1.2, 1.7596, 0.53506, 0.28052, 1, 1.8023, 0.0049535, 0, 0, 0.0099604],
            4: [13.69, 5.7188, 2.1, 0.86821, 0.31745, 0.54326, 1, 1.8001, 0.0027908, 0, 0, 0.010618],
            5: [12.614, 5.824, 2.5, 0.77371, 0.24497, 0.54333, 0.89465, 1.8213, 0.0032194, 0, 0, 0.0090301],
            6: [12.504, 5.8098, 4, 0.68611, 0.1416, 0.54691, 0.80544, 1.848, 0.003376, 0, 0, 0.0064593],
            7: [9.3021, 5.6064, 1.2, 1.2336, 0.38537, 0.30135, 0.71465, 1.8714, 0.0058542, 0, 0, 0.012318],
            8: [10.772, 6.5405, 2.2779, 0.9, 0.28689, 0.53743, 1, 1.8036, 0.0038101, 0, 0, 0.0099583],
            9: [10.189, 5.4436, 1.032, 2.5, 0.59201, 0.20652, 1, 1.8031, 0.0046147, 0, 0, 0.0015753],
            10:[9.4256, 5.0715, 1.0368, 2.5, 0.58688, 0.21124, 1, 1.8005, 0.0053255, 0, 0, 0.0057607],
            11:[10.45, 6.4825, 1, 2.3413, 0.56682, 0.20762, 0.92665, 1.8164, 0.0040848, 0, 0, 0.0032667],
        }
    },
    # =====================================
    "HA3_b_CA4_LV4_7p5aSyn": {
        "ellipsoid_asym": [10.615, 6.2472, 1.1741, 0.21722, 1.478,  1.7475,  0.36964,  0.20011,  0.73867, 1.7677, 0.00013644, 0.00052475, 0, 0.11457],
        "spheric_asym": {
            1: [8.4139, 5.1032, 1.0241, 2.0023, 0.55397, 0.24755, 0.91623, 1.8027, 0.0039824, 0, 0, 0.0077134],
            2: [7.4282, 5.9702, 1.0706, 1.5275, 0.51155, 0.29109, 0.84233, 1.8189, 0.0043685, 0, 0, 0.0094872],
            3: [7.3679, 6.3325, 1.1398, 2.4241, 0.59309, 0.2, 1, 1.7875, 0.0040715, 0, 0, 0.009387],
        }
    }, 
    "HA3_b_CA5_LV4_3aSyn": {
        "ellipsoid_asym": [10.497, 6.4909, 1.083, 0.24643, 1.4598, 1.7853,  0.34637,  0.20003,  0.7235,  1.8187, 0.00018308, 0.16149, 0.0017553, 0.35128],
        "spheric_asym": {
            1: [9.1349, 5.7547, 1.0908, 2.1, 0.57605, 0.2404, 0.99167, 1.7943, 0.0039734, 0, 0, 0.016092],
            2: [8.4144, 5.9348, 0.98514, 4, 0.59243, 0.11764, 0.91555, 1.8084, 0.004434, 0, 0, 0.013124],
            3: [10.899, 5.5984, 3.7658, 0.7236, 0.1665, 0.56637, 0.88475, 1.8177, 0.0031143, 0, 0, 0.014425],
            4: [12.67, 5.3028, 2.1, 0.9, 0.32592, 0.51857, 1, 1.7935, 0.0024381, 0, 0, 0.018304],
            5: [9.1599, 6.1014, 2.5, 0.9, 0.26179, 0.54579, 1, 1.7951, 0.0037973, 0, 0, 0.017024], 
            6: [8.4569, 5.6661, 1.4276, 1, 0.34985, 0.36708, 0.73285, 1.8542, 0.0050134, 0, 0, 0.019781],
            7: [9.5491, 4.9925, 1.0486, 2.5, 0.591, 0.20643, 1, 1.7929, 0.0040177, 0, 0, 0.0089747],
            8: [9.6308, 5.7975, 1.0928, 2.5, 0.57362, 0.2043, 1, 1.7947, 0.0036357, 0, 0, 0.0024374],
        }
    },
    "HA3_b_CA6_LV4_1aSyn": {
        "ellipsoid_asym": [9.3033, 6.0629, 1.1275, 0.11246, 1.2544, 1.5019,  0.35065,  0.20000,  0.608, 1.8441, 0.00021972, 0.28776, 6.3187e-11, 0.0035365],
        "spheric_asym": {
            1: [8.6316, 5.3679, 1.0434, 1.4189, 0.45956, 0.2626, 0.71478, 1.8677, 0.0043952, 0, 0, 0.0099543],
            2: [9.0218, 5.3657, 1.2, 1.3053, 0.47892, 0.33767, 0.86842, 1.8174, 0.0037735, 0, 0, 0.010297],
            3: [11.506, 4.9016, 1.7621, 0.77417, 0.29792, 0.43573, 0.71541, 1.8677, 0.0028883, 0, 0, 0.010248],
            4: [9.6235, 5.4441, 1.7612, 1, 0.37821, 0.48871, 1, 1.7943, 0.0032775, 0, 0, 0.010431],
            5: [8.7605, 5.4333, 1.4503, 0.9, 0.34319, 0.39167, 0.70876, 1.8674, 0.0043159, 0, 0, 0.010814],
            6: [8.8767, 4.7144, 1.0511, 1.6804, 0.52847, 0.28143, 0.87925, 1.8162, 0.004076, 0, 0, 0.013136],
            7: [9.8073, 5.5091, 1.0866, 2.0197, 0.55237, 0.2, 0.85188, 1.8195, 0.0033705, 0, 0, 0.0045747],
        }
    },
    "HA3_b_CA7_LV4_0p3aSyn": {
        "ellipsoid_asym": [9.8172, 6.0594, 1.1591, 0.15875, 1.2739, 1.7445,  0.36741,  0.20000,  0.68632, 1.8015, 0.00020625, 0.19825, 1.5828e-9, 0.080816],
        "spheric_asym": {
            1: [9.3626, 5.5579, 1.0569, 2.1, 0.5506, 0.2107, 0.8837, 1.8144, 0.004049, 0, 0, 0.008428],
            2: [8.7961, 5.6865, 0.98057, 4, 0.56709, 0.10567, 0.84026, 1.8265, 0.0044509, 0, 0, 0.0063177],
            3: [11.858, 5.1485, 2.1, 0.84894, 0.30009, 0.51784, 0.9229, 1.8062, 0.0028236, 0, 0, 0.010072],
            4: [10.457, 5.3833, 2.5, 0.78844, 0.24002, 0.53007, 0.87181, 1.8187, 0.0034489, 0, 0, 0.0090118],
            5: [9.2682, 5.625, 3.9975, 0.72845, 0.14161, 0.55113, 0.82511, 1.8315, 0.0041581, 0, 0, 0.0071106],
            6: [9.6719, 5.2616, 1.4499, 1, 0.36052, 0.37621, 0.76189, 1.846, 0.0041976, 0, 0, 0.011604],
            # 7: [25, 8, 0.94995, 1.8996, 0.51955, 0.26454, 0.87509, 1.8201, 0.00069042, 0, 0, 0.0093877],
            8: [9.1337, 5.1563, 1.0948, 2.5, 0.57738, 0.20521, 1, 1.7966, 0.0041632, 0, 0, 0.01317],
            9: [10.117, 5.4228, 1.1599, 2.3833, 0.57717, 0.2, 1, 1.7916, 0.0035706, 0, 0, 0.00032897],
        }
    }, 
    "HA3_b_CA8_LV4_0p1aSyn": {
        "ellipsoid_asym": [], # FAILED
        "spheric_asym": {
            1: [9.6039, 4.5263, 1.0213, 2.5, 0.6, 0.20927, 1, 1.7856, 0.0040659, 0, 0, 0.010744],
        }
    },
    # =================== SiO2 and CSLB ===================================================
    "HA1_CA4_1mgSiO2_60": {
        # chi2 = 14.799 (op. error)
        "hard_sphere_polydisp": [26.62, 1.72, 5.773e-07, 0.01944],
        # chi2 = 14.83 (op. error), fuzzyness is obviously not very relevant
        "rough_hard_sphere_polydisp": [27.22, 1.699, 4.112e-05, 1.434e-13, 0.01944],
        # chi2 = 14, notable improvement ==> ellipticity plays a role!
        "ellipsoid_homogen_polydisp": [26.92, 1.108, 0.9686, 0.1315, 5.746e-07, 0.01973],
        
        # ======== DAB ======================> DAB gives major improvements
        # chi2 = 10.2 
        "hard_sphere_polydisp_DAB": [26.67, 1.426, 4.834e-07, 0.01167, 4.503, 4.201],
        # chi2 = 8.8 
        "ellipsoid_homogen_polydisp_DAB": [26.76, 0.8199, 0.989, 0.1593, 5.112e-07, 0.003631, 0.5258, 2.008],
    },
    "HA3_CA11_CSLB_3_sncfB": {
        # chi2 = 46
        "hard_sphere_polydisp": [27.87, 1.63, 5.847e-08, 0.003136], # NOT SATISFYING
        # chi2 = 19.058
        "rough_hard_sphere_polydisp": [28.39, 1.675, 7.794e-05, 9.953e-14, 0.003686],
        # chi2 = 15.96
        "ellipsoid_homogen": [27.9, 0.7906, 1.006, 0.1722, 5.479e-08, 0.003848],
        # chi2 = 143 Feb20["R_fit","sigma_R_fit","z_h_fit","sigma_h_fit","rho_h_fit","z_c_fit","sigma_c_fit","rho_SiO2_fit","scale_fit", "c4_fit"] 
        "CSLB_spherical": [27.19,        3.223,       1.5,     0.2098,         1.241,   3.065,        0.4353,        593.5,  1.004e-16, 7.688e-05],
        # chi2 = 87
        "CSLB_ellipsoidal": [26.61, 2.11, 1.001, 0.01121, 1.843, 0.3015, 1.413, 3.001, 0.8416, 9.864e+04, 5.538e-18, -1.126e-05],
        # =========== DAB ============================
        # chi2 = 15 , but very bad fit !?
        "hard_sphere_polydisp_DAB": [],
        # chi2 = 19 , bad fit too
        "rough_hard_sphere_polydisp_DAB": [],
        # chi2 = 15.7 - Feb20      ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "scale_fit", "c4_fit", "c_fit", "xi_fit"] - BEST FIT
        "ellipsoid_homogen_DAB": [28.11,       0.3572,     0.9699,       0.1918,        5.093e-08, -0.03548, 0.05251, 0.4076],
        # chi2 = 35 , extremely bad fit
        "CSLB_spherical_DAB": [],
        # chi = 42 Feb23
        "CSLB_ellipsoidal_DAB": [28.11, 0.4101, 0.9706, 0.1752, 2.197, 0.6, 3.989, 2.372, 0.6726, 150.1, 2.276e-12, -0.01864, 0.04061, 0.6092],
    },
    # =================== CSLB + aSyn =====================================================
    "HA6_CA8_CSLB3_15uMaSyn": {
        "ellipsoid_homogen_DAB": [],
    },
    # ====================== LV2 ==========================================================
    "HA1_CA7_86mMLV2_60":{
        "flat_sym_restricted": [0.3431, 0.6, 1.092, 1.878, 0.48, 0, 0, 0.0173],
        "flat_asym_restricted": [0.2413, 0.5441, 1, 2.5, 1.052, 1.779, 0.358, 0, 0, 0.0133],
        "flat_asym": [0.2583, 0.5514, 0.9359, 2.5, 1.008, 1.8, 0.3391, 0, 0, 0.01305], 
        "spheric_sym": [8.17, 22.54, 0.4054, 0.8777, 1.401, 1.805, 0.0006138, 0, 0, 0.0159], # FAILED, obviously
        # chi2 = 4.15 # observation: fit is worse if bounds are more realistic
        "spheric_asym": [9.938, 6.165, 1.236, 2.2, 0.5741, 0.2493, 0.9755, 1.788, 0.002533, 0, 0, 0.01308],
        # chi2 = 4.55 (op error)
        "ellipsoid_asym": [10.6, 8.376, 0.9345, 0.1314, 1.419, 2.097, 0.5, 0.2635, 0.9737, 1.844, 3.88e-06, 0.01333],
    },
    # ======================== aSyn variants + LV4 =========================================
    # === G14R =====================
    "HA6_CA4_LV4_25uM_G14R": {
        "ellipsoid_asym": [9.117, 6.719, 0.9118, 0.4617, 5.059, 1.816, 0.1, 0.3527, 1, 1.78, 4.84e-06, 0, 0, 0.01284],
    },
    "HA6_CA5_LV4_15uM_G14R": {
        "ellipsoid_asym": [8, 5.854, 1.126, 0.2264, 1.896, 1.307, 0.2483, 0.3679, 0.8142, 1.776, 6.209e-06, 0, 0, 0.01495],
    },
    # === A30P ========================
    "HA6_CA6_LV4_25uM_A30P": {
        "ellipsoid_asym": [8.044, 7.513, 1.11, 0.2773, 3.402, 1.518, 0.1, 0.2825, 0.6608, 1.833, 8.736e-06, 0, 0, 0.01008],
    },
    "HA6_CA7_LV4_15uM_A30P": {
        "ellipsoid_asym": [9.915, 6.201, 1.102, 0.3351, 4, 1.477, 0.1219, 0.3725, 0.9093, 1.763, 6.451e-06, 0, 0, 0.00255],
    },
}


#============= function for chi2 and info_text ==================

def info(q, I, I_err, param_names, p0, popt, perr, function):

    residuals = I - function(q, *popt)
    chi2 = np.sum((residuals / I_err)**2) / (len(I) - len(p0) - 1)

    info_text = "\n".join(f"{name} = {val:.5g} ± {err:.5g}" for name, val, err in zip(param_names, popt, perr))
    info_text += f"\nchi2 = {chi2:.5g}"
    print("Fit results with 1σ uncertainties:\n")
    print(info_text)
    return chi2, info_text

def all_info(q, I, I_err, param_names, p0, popt, perr, function, lower_bounds, upper_bounds):

    residuals = I - function(q, *popt)
    chi2 = np.sum((residuals / I_err)**2) / (len(I) - len(p0) - 1)

    info_text = "\n".join(f"{name} = {val:.5g} ± {err:.5g}" for name, val, err in zip(param_names, popt, perr))
    info_text += f"\nchi2 = {chi2:.5g}"
    # print("Fit results with 1σ uncertainties:\n")
    # print(info_text)

    residuals = I - function(q, *p0)
    initial_chi2 = np.sum((residuals / I_err)**2) / (len(I) - len(p0) - 1)
    names2 = [name[:-4] for name in param_names]
    info_text2 = "\n".join(f"{name} = {val:.5g} ({(lower_bounds[i]):.3g}, {(upper_bounds[i]):.3g})" for name, val, i in zip(names2, p0, range(len(p0))))
    info_text2 += f"\nchi2_initial = {initial_chi2:.5g}\n \n"
    
    info_text_print = info_text2 + info_text

    return chi2, info_text_print

#============= plot together data ==================

import itertools

def plot_together(who, title, version, target_folder=None, background=False):
    
    # Plot SAXS data for given samples in RAW, SUB, RAW_normalized, SUB_normalized, or RAW,SUB mode.

    # Parameters
    # ----------
    # who : list[str]
    #     List of sample names (without '_SAXS.dat').
    # title : str
    #     Plot title and filename.
    # version : str
    #     "RAW", "SUB", "RAW_normalized", "SUB_normalized", or "RAW,SUB"
    # target_folder : Path or None, optional
    #     Directory to save the figure, by default None.
    # background : bool, optional
    #     If True, plot background once per top_folder in RAW plots (each with distinct linestyle).
    

    plt.close("all")
    who = [sample_name + "_SAXS.dat" for sample_name in who]

    # ---- Define linestyle cycle for background ----
    line_styles = itertools.cycle(["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))])

    # ---- Helper: plot background if available ----
    def plot_background(ax, top_folder, linestyle):
        if background and top_folder in backgrounds:
            q_bgr, I_bgr, I_bgr_err = np.loadtxt(backgrounds[top_folder], unpack=True)
            if backgrounds[top_folder].stem[:-5].endswith("60"):
                I_bgr = I_bgr / 60
            else:
                I_bgr = I_bgr / 30
            ax.loglog(
                q_bgr, I_bgr,
                color="black",
                linestyle=linestyle,
                linewidth=1.5,
                label=f"bgr ({backgrounds[top_folder].stem[:-5]})"
            )
    
    # ---- Helper: find first local maximum ----
    def first_peak_height(q, I):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(I)
        first = peaks[0]
        if first > 30:
            return None, I[0]

        return q[first], I[first]

    # ---- Single panel (RAW, SUB, RAW_normalized, SUB_normalized) ----
    if version in ("RAW", "SUB", "RAW_normalized", "SUB_normalized"):
        fig, ax = plt.subplots(figsize=(12, 7.5))
        plotted_bgr = {}
        normalization_factor = None

        for sample_name in who:
            index = sample_name.find('C')
            top_folder = sample_name[:index - 1]
            # determine subfolder dynamically
            subfolder = "SAXS/SUB" if "SUB" in version else "SAXS/RAW"
            sample_path = base_path / top_folder / sample_name[:-9] / subfolder / sample_name

            bgr_key = get_background_key(top_folder, sample_name)

            if version.startswith("RAW") and background and bgr_key not in plotted_bgr:
                linestyle = next(line_styles)
                plot_background(ax, bgr_key, linestyle)
                plotted_bgr[bgr_key] = linestyle


            q, I, I_err = np.loadtxt(sample_path, unpack=True)
            if sample_name[:-9].endswith("60"):
                I = I / 60
            else:
                I = I / 30

            # --- Normalize by first maximum intensity (for *_normalized) ---
            if version.endswith("normalized"):
                # _, Inorm = first_peak_height(q, I)
                #Inorm = I[0]
                closest_index = np.argmin(np.abs(q - 0.1))
                Inorm = I[closest_index]
                if Inorm is not None:
                    if normalization_factor is None:
                        normalization_factor = Inorm  # first curve defines reference
                    I = I * (normalization_factor / Inorm)

            ax.loglog(q, I, label=sample_path.stem[:-5])

        # Axis limits
        if "SUB" in version:
            ax.set_xlim(0.05, 6)
            ax.set_ylim(1e-2, None)

        ax.set_xlabel("$q[1/\\mathrm{nm}]$")
        ax.set_ylabel("$I(q)\\Bigl{[}\\frac{\\mathrm{ph}}{\\mathrm{px}\\cdot\\mathrm{s}}\\Bigr{]}$")
        title_full = title
        if version.endswith("normalized"):
            title_full += " (normalized)"
        ax.set_title(title_full)
        ax.grid(True)
        ax.legend(fontsize=12)
        fig.tight_layout()

        if target_folder is not None:
            target_folder = Path(target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_folder / f"{title}.png", dpi=600, bbox_inches='tight', pad_inches=0.05)

        plt.show()
        return

    # ---- Two-panel version (RAW,SUB) ----
    if version == "RAW,SUB":
        fig, axes = plt.subplots(1, 2, figsize=(22, 8), dpi=1000)
        plotted_bgr = {}

        for ax, subver in zip(axes, ["RAW", "SUB"]):
            for sample_name in who:
                index = sample_name.find('C')
                top_folder = sample_name[:index - 1]
                subfolder = f"SAXS/{subver}"
                sample_path = base_path / top_folder / sample_name[:-9] / subfolder / sample_name

                bgr_key = get_background_key(top_folder, sample_name)

                if subver == "RAW" and background and bgr_key not in plotted_bgr:
                    linestyle = next(line_styles)
                    plot_background(ax, bgr_key, linestyle)
                    plotted_bgr[bgr_key] = linestyle


                q, I, I_err = np.loadtxt(sample_path, unpack=True)
                if sample_name[:-9].endswith("60"):
                    I = I / 60
                else:
                    I = I / 30
                ax.loglog(q, I, label=sample_path.stem[:-5])

            if subver == "SUB":
                ax.set_xlim(0.05, 6)
                ax.set_ylim(1e-2, None)

            ax.set_xlabel("$q[1/\\mathrm{nm}]$")
            ax.set_ylabel(r"$I(q) \\Bigl{[}\\frac{\\mathrm{ph}}{\\mathrm{px}\\cdot\\mathrm{s}}\\Bigr{]}$")
            ax.set_title(subver)
            ax.grid(True)
            ax.legend(fontsize=12)

        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if target_folder is not None:
            target_folder = Path(target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_folder / f"{title}.png", dpi=700, bbox_inches='tight', pad_inches=0.05)

        plt.show()
        return

#============== Get RAW or SUB data ============================================
# sample name of the form "HA3_c_CA6_LV4_1aSyn"

def get_data(sample_name, version):
    ID10_path = Path("/home/AG_Salditt/Projekte_Membranbiophysik/Riko_BA/Data/ID10-COH_2024_SC5574/processed_Andras/")
    special_sample = sample_name + ".dat"
    
    sample_name = sample_name + "_SAXS.dat"
    index = sample_name.find('C')
    top_folder = sample_name[:index-1] 
    if version == "SUB":
        sample_path = base_path / top_folder / sample_name[:-9] / "SAXS/SUB/" / sample_name
    elif version == "RAW":
        if special_sample.startswith("SAXS"):
            sample_path = ID10_path / special_sample
            q, I = np.loadtxt(sample_path, unpack=True)
            return q, I, np.zeros(len(I))
        if ("CSLB" in sample_name) and ("zscan" in sample_name):
            sample_path = base_path / "SPECIAL" / special_sample
        else:
            sample_path = base_path / top_folder / sample_name[:-9] / "SAXS/RAW/" / sample_name
    elif version == "INT":
        sample_path = base_path / top_folder / sample_name[:-9] / "SAXS/INT/" / sample_name
    elif version == "SPECIAL":
        sample_path = base_path / "SPECIAL" / special_sample
    q, I, I_err = np.loadtxt(sample_path, unpack=True)
    return q, I, I_err

#========== Backgrounds for each HA ============================
def get_background_key(top_folder, sample_name):
    # top_folder e.g. "HA5"
    # sample_name e.g. HA1_CA3_TBS_SAXS.dat" or "HA1_CA3_TBS"
    name_no_suffix = sample_name
    if sample_name.endswith(".dat"):
        name_no_suffix = sample_name[:-9]   # remove '_SAXS.dat'
    
    
    if name_no_suffix.endswith("CaCl2") or name_no_suffix.endswith("LV2"):
        return top_folder + "_MQ"
    if name_no_suffix.endswith("CaCl2_60") or name_no_suffix.endswith("LV2_60"):
        return top_folder + "_MQ_60"
    if name_no_suffix.endswith("60"):
        return top_folder + "_TBS_60"
    if top_folder.startswith("HA4"):
        return top_folder
    else:
        return top_folder + "_TBS"

backgrounds = {
    "AgBh": base_path / "Air/Air_empty/SAXS/RAW/" / "Air_empty_SAXS.dat",   # not really useful
    "Air": base_path / "Air/Air_empty/SAXS/RAW/" / "Air_empty_SAXS.dat",    # not really useful
    
    "HA1_TBS": base_path / "HA1/HA1_CA3_TBS/SAXS/RAW/" / "HA1_CA3_TBS_SAXS.dat",
    "HA1_TBS_60": base_path / "HA1/HA1_CA3_TBS_60/SAXS/RAW/" / "HA1_CA3_TBS_60_SAXS.dat",
    "HA1_MQ": base_path / "HA1/HA1_CA2_MQ/SAXS/RAW/" / "HA1_CA2_MQ_SAXS.dat",
    "HA1_MQ_60": base_path / "HA1/HA1_CA2_MQ_60/SAXS/RAW/" / "HA1_CA2_MQ_60_SAXS.dat",
    
    "HA1_test_TBS_60": base_path / "HA1/HA1_CA3_TBS_60/SAXS/RAW/" / "HA1_CA3_TBS_60_SAXS.dat",

    "HA2_TBS": base_path / "HA2/HA2_CA3_TBS/SAXS/RAW/" / "HA2_CA3_TBS_SAXS.dat",
    "HA2_TBS_60": base_path / "HA2/HA2_CA3_TBS_60/SAXS/RAW/" / "HA2_CA3_TBS_60_SAXS.dat",
    "HA2_MQ": base_path / "HA2/HA2_CA2_MQ/SAXS/RAW/" / "HA2_CA2_MQ_SAXS.dat",
    "HA2_MQ_60": base_path / "HA2/HA2_CA2_MQ_60/SAXS/RAW/" / "HA2_CA2_MQ_60_SAXS.dat",
    
    "HA2_b_TBS": base_path / "HA2/HA2_CA3_TBS_60/SAXS/RAW/" / "HA2_CA3_TBS_60_SAXS.dat",
    "HA2_b_TBS_60": base_path / "HA2/HA2_CA3_TBS_60/SAXS/RAW/" / "HA2_CA3_TBS_60_SAXS.dat",
    
    "HA3_TBS": base_path / "HA3/HA3_CA3_TBS/SAXS/RAW/" / "HA3_CA3_TBS_SAXS.dat",
    "HA3_1_TBS": base_path / "HA3/HA3_CA3_TBS/SAXS/RAW/" / "HA3_CA3_TBS_SAXS.dat", # not really useful
    "HA3_b_TBS": base_path / "HA3_b/HA3_b_CA3_TBS/SAXS/RAW/" / "HA3_b_CA3_TBS_SAXS.dat",
    "HA3_c_TBS": base_path / "HA3_c/HA3_c_CA3_TBS/SAXS/RAW/" / "HA3_c_CA3_TBS_SAXS.dat",
    
    "HA4": base_path / "HA4/HA4_CA2_MQ/SAXS/RAW/" / "HA4_CA2_MQ_SAXS.dat",              # only MQ
    "HA4_b": base_path / "HA4_b/HA4_b_CA2_MQ/SAXS/RAW/" / "HA4_b_CA2_MQ_SAXS.dat",      # only MQ
    "HA4_c": base_path / "HA4_c/HA4_c_CA2_MQ/SAXS/RAW/" / "HA4_c_CA2_MQ_SAXS.dat",      # only MQ
    "HA4_d": base_path / "HA4_d/HA4_d_CA2_MQ/SAXS/RAW/" / "HA4_d_CA2_MQ_SAXS.dat",      # only MQ
    
    "HA5_TBS": base_path / "HA5/HA5_CA3_TBS/SAXS/RAW/" / "HA5_CA3_TBS_SAXS.dat",
    
    "HA6_TBS": base_path / "HA6/HA6_CA3_TBS/SAXS/RAW/" / "HA6_CA3_TBS_SAXS.dat",
    "HA6_MQ": base_path / "HA6/HA6_CA2_MQ/SAXS/RAW/" / "HA6_CA2_MQ_SAXS.dat",
}

subtraction_factors = {
    # LV4 pure
    "HA1_CA6_30mMLV4": 1.05, 
    "HA1_CA6_30mMLV4_60": 1.03,
    "HA3_CA9_LV4a": 1.06,  # the fit kept me busy for months
    "HA3_CA10_LV4b": 1.08,
    "HA3_b_CA9_LV4a": 1.03,
    "HA3_b_CA10_LV4b": 1.08,
    "HA3_c_CA9_LV4a": 1.04,
    "HA3_c_CA10_LV4b": 1.075,

    # LV4 + alpha-Synuclein
    "HA5_CA6_LV4_9uMaSyn": 1.07,
    "HA5_CA4_LV4_15uMaSyn": 1.03,
    "HA5_CA5_LV4_25uMaSyn": 1.1,

    "HA3_CA4_LV4_7p5aSyn": 0.98,     
    "HA3_b_CA4_LV4_7p5aSyn": 0.9887, 
    "HA3_c_CA4_LV4_7p5aSyn": 0.98, 

    "HA3_CA5_LV4_3aSyn": 1.06,
    "HA3_b_CA5_LV4_3aSyn": 1.005,
    "HA3_c_CA5_LV4_3aSyn": 1.04,
    
    "HA3_CA6_LV4_1aSyn": 1.02,
    "HA3_b_CA6_LV4_1aSyn": 1.005,
    "HA3_c_CA6_LV4_1aSyn": 1.006,

    "HA3_CA7_LV4_0p3aSyn": 1.04,
    "HA3_b_CA7_LV4_0p3aSyn": 1.03,
    "HA3_c_CA7_LV4_0p3aSyn":  1.03,

    "HA3_CA8_LV4_0p1aSyn": 1.04,    
    "HA3_b_CA8_LV4_0p1aSyn": 1.0,
    "HA3_c_CA8_LV4_0p1aSyn": 1.01,

    "HA3_CA11_CSLB_3_sncfB": 1.57,
    "HA6_CA8_CSLB3_15uMaSyn": 1.125,

    # LV4 + aSyn + Syn1
    "HA6_CA4_LV4_1uMSyn_2uMaSyn": 1.05,
    "HA5_CA11_LV4_2p5uMSyn_5uMaSyn": 1.03,   
    "HA5_CA9_LV4_3uMSyn_6uMaSyn": 1.01,
    
    "HA5_CA13_LV4_2uMSyn_1uMaSyn": 1.01, 
    "HA5_CA10_LV4_5uMSyn_2p5uMaSyn": 1.06,
    "HA5_CA8_LV4_6uMSyn_3uMaSyn": 1.00,
    
    "HA5_CA7_LV4_6uMSyn": 1.0,

    # SV + aSyn + Syn1
    "HA6_CA6_23nMSV_3uMSyn_6uMaSyn": 0.95,
    "HA6_CA7_23nMSV_6uMSyn_3uMaSyn": 1.0,
    "HA6_CA7_23nMSV_6uMSyn_3uMaSyn_2": 0.9,
                                       
    "HA2_CA12_23nMSV_2": 0.95,
    "HA2_CA12_23nMSV_60": 0.925, 
    "HA2_b_CA12_23nMSV_60": 0.95,
}

# ==== function to calculate subtraction factors automatically ================================

def calculate_sub_factor(q_RAW, I_RAW, I_bgr):
    # define region of interest:
    q_mask    = q_RAW > 4.2
    I_sec     = np.copy(I_RAW[q_mask])
    I_bgr_sec = np.copy(I_bgr[q_mask])

    # filter out outliers:
    std_max = 0.1
    while np.std(I_sec) > std_max:
        deviation = []
        for I in I_sec:
            deviation.append(np.abs(I_sec.mean() - I))
        I_sec[np.argmax(deviation)] = I_sec[np.argmax(deviation)-1]  
        # print("replaced")
    while np.std(I_bgr_sec) > std_max:
        deviation = []
        for I in I_bgr_sec:
            deviation.append(np.abs(I_bgr_sec.mean() - I))
        I_bgr_sec[np.argmax(deviation)] = I_bgr_sec[np.argmax(deviation)-1]

    # slowly increase sub factor:
    dead_points = 1
    sub_factor = 0.9
    while np.sum(I_RAW - sub_factor * I_bgr <= 0) <= dead_points: 
        sub_factor += 0.025
        if np.sum(I_RAW - sub_factor * I_bgr <= 0) > dead_points:
            sub_factor -= 0.025
            break
    return sub_factor
    
#============= Param names for different fit functions ==========

param_names = {
    "spheric_sym" : ["R_fit", "sigma_R_fit", "sigma_head_fit", "sigma_tail_fit", 
                     "rho_head_fit", "eps_head_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "spheric_asym": ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", 
                     "sigma_h2_fit", "sigma_c_fit", "z_h_fit", "scale_fit", 
                     "c2_fit", "c3_fit", "c4_fit"],
    "flat_sym"    : ["sigma_head_fit", "sigma_tail_fit", "rho_head_fit", "eps_head_fit", 
                     "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "flat_sym_CONST_BGR": ["sigma_head_fit", "sigma_tail_fit", "rho_head_fit", "eps_head_fit", 
                           "scale_fit", "c4_fit"],
    "flat_sym_NO_BGR": ["sigma_head_fit", "sigma_tail_fit", "rho_head_fit", "eps_head_fit", "scale_fit"],
    "flat_asym"   : ["sigma_h1_fit", "sigma_h2_fit", "sigma_tail_fit", "rho_h1_fit", "rho_h2_fit", 
                     "eps_head_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "flat_asym_no_powlaw": ["sigma_h1_fit", "sigma_h2_fit", "sigma_tail_fit", "rho_h1_fit", "rho_h2_fit", 
                     "eps_head_fit", "scale_fit", "c4_fit"],
    "flat_asym_single_free_shell": ["sigma_h2_fit", "rho_h2_fit", "eps_head_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "ULV_ellipsoid": ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h_fit", 
                  "sigma_h_fit","sigma_c_fit","z_h_fit","scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "ULV_ellipsoid_asym": ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h1_fit", "rho_h2_fit",
                  "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit","scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "ellipsoid_homogen": ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "scale_fit"], 
    "CSLB_polydisp_NO_BGR": ["R_fit", "sigma_R_fit", "z_h_fit", "sigma_h_fit", 
                             "rho_h_fit", "z_c_fit", "sigma_c_fit", "rho_c_fit", "rho_SiO2_fit", "scale_fit"],
    "hard_sphere_polydisp": ["R_fit", "sigma_R_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "rough_hard_sphere_polydisp": ["R_fit", "sigma_R_fit", "a_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "hard_sphere_polydisp_NO_BGR": ["R_fit", "sigma_R_fit", "scale_fit"],
    "I_CL": ["scale_fit", "sigma_fit", "mu_fit", "lam_fit", "m_fit", "b_fit"], 
    "ULV_ell_asym_prot_layer": ["R_fit", "sigma_R_fit", "eps_fit", "sigma_eps_fit", "rho_h1_fit", "rho_h2_fit", "sigma_h1_fit", "sigma_h2_fit",
                                "sigma_c_fit","z_h_fit", "z_h3_fit", "sigma_h3_fit", "rho_h3_fit", "scale_fit", "c2_fit", "c3_fit", "c4_fit"],
    "ULV_sph_asym_prot_layer": ["R_fit", "sigma_R_fit", "rho_h1_fit", "rho_h2_fit",
                                "sigma_h1_fit", "sigma_h2_fit", "sigma_c_fit","z_h_fit", "z_h3_fit", "sigma_h3_fit", "rho_h3_fit", "scale_fit",
                                "c2_fit", "c3_fit", "c4_fit"]
}

#====================== Parameters for different cases and boundaries ==================================
"""
flat asym EPS ASYM
p0           = [sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2,   eps_h1,   eps_h2,   scale,      c2,      c3,      c4]
lower_bounds = [       0,        0,        0.5,    0.5,    0.5,        0,        0, -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [     1.5,      1.5,          2,      2,      2,        3,        3, +np.inf,    1e-2, +np.inf, +np.inf]

spheric asym EPS ASYM
p0           = [   R,    sigma_R, sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2,   eps_h1,   eps_h1,   scale,      c2,      c3,      c4]
lower_bounds = [  10,          1,        0,        0,        0.5,      0,      0,      0.5,      0.5, -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [  50,          9,      1.5,      1.5,          2,     10,     10,        3,        3, +np.inf,    1e-2, +np.inf, +np.inf]

spheric asym 
p0           = [   R,    sigma_R,   rho_h1, _h2, sigma_tail, rho_h1, rho_h2,   eps_head,   scale,      c2,      c3,      c4]
lower_bounds = [  10,          1,        0,        0,        0.5,      0,      0,        0.5, -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [  50,          9,      1.5,      1.5,          2,     10,     10,          3, +np.inf,    1e-2, +np.inf, +np.inf]

spheric sym
p0           = [   R,    sigma_R, sigma_head, sigma_tail, rho_head,  eps_head,    scale,      c2,      c3,      c4]
lower_bounds = [  10,          1,          0,        0.5,        0,       0.5,  -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [  50,          9,        1.5,          2,       10,         3,  +np.inf,    1e-2, +np.inf, +np.inf]

flat sym
p0           = [   sigma_head, sigma_tail, rho_head,  eps_head,    scale,      c2,      c3,      c4]
lower_bounds = [            0,        0.5,        0,       0.5,  -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [          1.5,          2,       10,         3,  +np.inf,    1e-2, +np.inf, +np.inf]

flat_asym
p0           = [   sigma_h1, sigma_h2, sigma_tail, rho_h1, rho_h2, eps_head,   scale,      c2,      c3,      c4]
lower_bounds = [        0.3,      0.3,        0.5,    0.8,    0.8,      1.5, -np.inf,       0, -np.inf, -np.inf]
upper_bounds = [        0.4,      0.4,          2,    1.6,    1.6,        2, +np.inf, +np.inf, +np.inf, +np.inf]

ULV_ellipsoid
p0           = [        R,  sigma_R,     eps, sigma_eps,  rho_h,     sigma_h,  sigma_c,      z_h,   scale,      c4]
lower_bounds = [      0.3,      0.3,     0.5,       0.8,    0.8,     -np.inf,        0,  -np.inf, -np.inf, -np.inf]
upper_bounds = [      0.4,      0.4,       2,       1.6,    1.6,     +np.inf,  +np.inf,  +np.inf, +np.inf, +np.inf]
ULV_ellipsoid_asym
p0           = [        R,  sigma_R,     eps, sigma_eps,  rho_h1, rho_h2,   sigma_h1, sigma_h2,  sigma_c,      z_h,    scale,   c2,   c3,       c4]
lower_bounds = [       12,        2,     0.8,         0,       1,      1,        0.2,      0.2,      0.5,      1.5,  -np.inf,    0,    0,  -np.inf]
upper_bounds = [       20,        9,     1.2,       0.7,     1.6,    1.6,        0.5,      0.5,        1,        2,  +np.inf,  1.5,  0.5,  +np.inf]

I_CL q, scale, sigma, mu, lam, m, b
p0           = [   scale, sigma,    mu,   lam,        m,       b]
lower_bounds = [       0,   0.5,     0,   0.5,  -np.inf,       0]
upper_bounds = [     1.5,     2,    10,     3,  +np.inf,    1e-2]
"""
# =======================================================================================================


#============== Komorowski 2018 best config LV4 ============================================================

"""
R_fit       = 36
sigma_R_fit = 13.8

sigma_head_fit = 0.46
sigma_tail_fit = 0.85

rho_head_fit = 1.09
eps_head_fit = 1.8

scale_fit = 
c2_fit    = 
c3_fit    = 
c4_fit    = 

"""

#============== Neuhaus 2024 best config LV4 ====================
"""
R_fit       = 13.85
sigma_R_fit = 8.1

sigma_head_fit = 0.34
sigma_tail_fit = 0.74

rho_head_fit = 1.42
eps_head_fit = 1.99

scale_fit = 1.3e-6
c2_fit    = 7.4e-9
c3_fit    = 5.2
c4_fit    = 1.6e-4

"""

#======================= Average Transmittance Coefficients for each Sample =======================

transmittance = {
    
    'MASK_SAX': None,
    'HA2_b_CA13_23nMSV_3uMaSyn_60': 0.931921,
    'HA2_b_CA12_23nMSV_60': 0.955073,
    'HA2_b_CA6_6mMLV4_3uMaSyn_60': 0.895962,
    'HA2_b_CA7_3mMLV4_3uMaSyn_60': 0.865448,
    
    'HA4_d_CA1_air': None,
    'HA4_d_CA9_20mM': None,
    'HA4_d_CA2_MQ': None,
    'HA4_d_CA8_10mM': None,
    'HA4_d_CA6_2mM': None,
    'HA4_d_CA11_100mM': None,
    'HA4_d_CA3_TBS': None,
    'HA4_d_CA7_5mM': None,
    'HA4_d_CA10_50mM': None,
    'HA4_d_CA4_0mM': None,
    'HA4_d_CA5_1mM': None,
    'HA4_d_CA13_500mM': None,
    'HA4_d_CA12_200mM': None,
    
    'HA4_c_CA5_1mM': 0.968333,
    'HA4_c_CA4_0mM': 0.953427,
    'HA4_c_CA13_500mM': 0.840514,
    'HA4_c_CA12_200mM': 0.747863,
    'HA4_c_CA7_5mM': 0.960360,
    'HA4_c_CA6_2mM': 1.017027,
    'HA4_c_CA8_10mM': 0.918054,
    'HA4_c_CA3_TBS': 0.950232,
    'HA4_c_CA2_MQ': 0.874970,
    'HA4_c_CA10_50mM': 0.921854,
    'HA4_c_CA9_20mM': 0.958065,
    'HA4_c_CA1_air': 1.647868,
    'HA4_c_CA11_100mM': 0.880376,
    
    'HA3_b_CA3_TBS': 0.965631,
    'HA3_b_CA8_LV4_0p1aSyn': 0.957185,
    'HA3_b_CA9_LV4a': 0.901105,
    'HA3_b_CA5_LV4_3aSyn': 0.937086,
    'HA3_b_CA1_air': 1.647486,
    'HA3_b_CA4_LV4_7p5aSyn': 0.954778,
    'HA3_b_CA2_MQ': 0.878100,
    'HA3_b_CA6_LV4_1aSyn': 0.934910,
    'HA3_b_CA7_LV4_0p3aSyn': 0.924504,
    'HA3_b_CA10_LV4b': 0.901617,
    
    'HA3_CA5_LV4_3aSyn': 0.935471,
    'HA3_CA8_LV4_0p1aSyn': 0.958166,
    'HA3_CA3_TBS': 0.962266,
    'HA3_CA12_CSLB_28a_sncf2': 0.899829,
    'HA3_CA2_MQ': 0.874749,
    'HA3_CA7_LV4_0p3aSyn': 0.922809,
    'HA3_CA6_LV4_1aSyn': 0.930945,
    'HA3_CA4_LV4_7p5aSyn': 0.954978,
    'HA3_CA10_LV4b': 0.900867,
    'HA3_CA9_LV4a': 0.900861,
    'HA3_CA11_CSLB_3_sncfB': 0.939474,
    
    'AgBh_3m_13keV_flow': None,
    
    'HA3_1_CA1_air': 1.641013,
    
    'HA4_CA12_200mM': 1.512998,
    'HA4_CA6_2mM': 1.471761,
    'HA4_CA10_50mM': 1.471938,
    'HA4_CA13_500mM': 0.837478,
    'HA4_CA2_MQ': 0.874980,
    'HA4_CA3_TBS': 0.924140,
    'HA4_CA9_20mM': 1.515799,
    'HA4_CA1_air': 1.648534,
    'HA4_CA13_500mM_2': 0.829495,
    'HA4_CA8_10mM': 1.500001,
    'HA4_CA5_1mM': 1.493774,
    'HA4_CA11_100mM': 1.484588,
    'HA4_CA7_5mM': 1.498039,
    'HA4_CA4_0mM': 1.498826,
    
    'HA6_CA9_500mMCaCl2': 0.500799,
    'HA6_CA8_CSLB3_15uMaSyn': 0.943424,
    'HA6_CA13_air': 1.645420,
    'HA6_CA4_LV4_25uM_G14R_2': None,
    'HA6_CA1_emptyCap': 1.487290,
    'HA6_CA6_LV4_25uM_A30P_2': None,
    'HA6_CA6_23nMSV_3uMSyn_6uMaSyn': 0.966710,
    'HA6_CA7_23nMSV_6uMSyn_3uMaSyn_2': 1.015720,
    'HA6_CA12_50mMCaCl2': 0.954474,
    'HA6_CA5_CSLB2SN': 0.983298,
    'HA6_CA7_LV4_15uM_A30P': 0.917370,
    'HA6_CA4_LV4_25uM_G14R': 0.918417,
    'HA6_CA3_TBS': 0.924526,
    'HA6_CA6_LV4_25uM_A30P': 0.868095,
    'HA6_CA4_LV4_1uMSyn_2uMaSyn': 0.832610,
    'HA6_CA5_LV4_15uM_G14R': 0.923888,
    'HA6_CA10_200mMCaCl2': 0.745874,
    'HA6_CA11_100mMCaCl2': 0.893263,
    'HA6_CA7_23nMSV_6uMSyn_3uMaSyn': 0.938450,
    'HA6_CA2_MQ': 0.863952,
    
    'HA4_b_CA2_MQ': None,
    'HA4_b_CA1_air': None,
    'HA4_b_CA3_TBS': None,
    'HA4_b_CA4_0mM': 1.500498,
    
    'HA3_c_CA4_LV4_7p5aSyn': 0.953941,
    'HA3_c_CA5_LV4_3aSyn': 0.937504,
    'HA3_c_CA7_LV4_0p3aSyn': 0.924580,
    'HA3_c_CA10_LV4b': 0.902509,
    'HA3_c_CA3_TBS': 0.964286,
    'HA3_c_CA2_MQ': 0.876837,
    'HA3_c_CA6_LV4_1aSyn': 0.933290,
    'HA3_c_CA1_air': 1.643743,
    'HA3_c_CA9_LV4a': 0.902159,
    'HA3_c_CA8_LV4_0p1aSyn': 0.957809,
    
    'Air_empty': None,
    
    'HA1_CA7_86mMLV2_60': 0.951212,
    'HA1_CA1_air_60': 1.628964,
    'HA1_CA6_30mMLV4': 0.887401,
    'HA1_CA3_TBS_60': 0.876678,
    'HA1_CA2_MQ_60': 0.859239,
    'HA1_CA4_1mgSiO2_60': 0.913968,
    'HA1_CA2_MQ': 0.855089,
    'HA1_CA5_01mgSiO2_60': 0.934833,
    'HA1_CA4_1mgSiO2': 0.908941,
    'HA1_CA1_air': 1.620937,
    'HA1_CA6_30mMLV4_60': 0.890776,
    'HA1_CA5_01mgSiO2': 0.931890,
    'HA1_CA3_TBS': 0.872037,
    'HA1_CA7_86mMLV2': 0.949173,
    
    'HA5_CA2_MQ': 0.896069,
    'HA5_CA12_CSLB3_2uMSyn_1uMaSyn': 0.963779,
    'HA5_CA5_LV4_25uMaSyn': 0.859772,
    'HA5_CA11_LV4_2p5uMSyn_5uMaSyn': 0.926415,
    'HA5_CA7_LV4_6uMSyn': 0.945491,
    'HA5_CA6_LV4_9uMaSyn': 0.885626,
    'HA5_CA12_CSLB3_2uMSyn_1uMaSyn_zscan': 1.018865,
    'HA5_CA1_air': 1.645498,
    'HA5_CA13_LV4_2uMSyn_1uMaSyn': 0.940986,
    'HA5_CA9_LV4_3uMSyn_6uMaSyn': 0.950037,
    'HA5_CA4_LV4_15uMaSyn': 0.944766,
    'HA5_CA8_LV4_6uMSyn_3uMaSyn': 0.929324,
    'HA5_CA10_LV4_5uMSyn_2p5uMaSyn': 0.859430,
    'HA5_CA7_LV4_6uMSyn_zscan': 0.986627,
    'HA5_CA3_TBS': 0.969157,
    
    'HA1_test_LV2': None,
    
    'HA2_CA2_MQ_60b': None,
    'HA2_CA10_5mMLV2_10mMCaCl2_60': 0.881034,
    'HA2_CA7_3mMLV4_3uMaSyn': 0.859147,
    'HA2_CA7_3mMLV4_3uMaSyn_60': 0.862623,
    'HA2_CA1_air_60b': None,
    'HA2_CA13_23nMSV_3uMaSyn_60': 0.928908,
    'HA2_CA1_air': 1.628300,
    'HA2_CA2_MQ': 0.858520,
    'HA2_CA3_TBS': 0.876917,
    'HA2_CA6_6mMLV4_3uMaSyn_60b': None,
    'HA2_CA12_23nMSV_60': 0.953614,
    'HA2_CA12_23nMSV_2': 0.948737,
    'HA2_CA2_MQ_60': 0.860793,
    'HA2_CA12_23nMSV': 1.056312,
    'HA2_CA3_TBS_60': 0.881401,
    'HA2_CA10_5mMLV2_10mMCaCl2_60b': None,
    'HA2_CA3_TBS_60b': None,
    'HA2_CA12_23nMSV_60b': None,
    'HA2_CA6_6mMLV4_3uMaSyn': 0.889731,
    'HA2_CA1_air_60': 1.632536,
    'HA2_CA13_23nMSV_3uMaSyn_60b': None,
    'HA2_C13_23nMSV_3uMaSyn': 0.925044,
    'HA2_CA6_6mMLV4_3uMaSyn_60': 0.893510,
    'HA2_CA10_5mMLV2_10mMCaCl2': 0.877482,
    'HA2_CA7_3mMLV4_3uMaSyn_60b': None
}






