import fit_lib as fit
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = False
import re

def extract_sample_name(sample):
    sample = sample[:-13] + "_"
    match = re.search(r"sample_([^_]+_[^_]+)", sample)
    result = match.group(1)
    block = result.split("_")[0]
    return result, block

def blockpath(block):
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent      
    data_root = project_root / "data_BM29_July2026"
    
    return data_root / block

def plot_together(who, title):
    """
    who: list of samples of the form "BII_S3"
    """
    plt.close("all")
    
    f = plt.figure(figsize=(12, 7.5))
    for sample in who:
        block = sample.split("_")[0]
        block_path = blockpath(block)
        dir_names = [x.name for x in block_path.iterdir()]
        for dir_name in dir_names:
            if dir_name[:-13].endswith(sample):
                sub_dat_path = block_path / dir_name / "avg_and_sub" / sample_dict[sample]
                q, I, Ierr = np.loadtxt(sub_dat_path, unpack=True)
                plt.errorbar(q, I, Ierr, ls = "None", marker="o", label=sample_dict[sample], markersize=2.0)
    plt.xlabel("$q$")
    plt.ylabel("$I$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), alignment='left')
    return

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
    "BIX_S19": "BIX_S19_15mM_DOPC_DOPS_80_20_50uM_aSynWT"
}

best_fits = {
    'BII_S9': {
        "flat_asym_C4_q0p7": {
            #  sigma_h1, sigma_h2, sigma_tail, rho_h1,  rho_h2,   eps_head,       scale,     c4
            1: [0.2481,    0.3061,     0.6217,      1,   1.463,      1.908,      0.0488, 0.000501]
        },
        'ULV_spheric_asym_C4': {
            1: [16.09, 2.936, 1.383, 0.9795, 0.3527, 0.3049, 0.6648, 1.907, 0.0001727, 0.0005608]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.89, 2.825, 0.8596, 1.468e-08, 1.473, 1.079, 0.3535, 0.2949, 0.716, 1.989, 3.957e-07, 0.0004725],
            # 2: [13.19, 4.587, 1.081, 0.4, 2.9, 2.509, 0.1307, 0.1, 0.5185, 1.87, 5.487e-07, 0.0002264],
            # 3: [16.16, 3.131, 1.133, 0.2141, 1.381, 1.035, 0.3428, 0.2975, 0.657, 1.807, 3.14e-07, 0.000499],
            # 4: [16.26, 3.134, 1.09, 0.2514, 1.388, 1.073, 0.343, 0.2949, 0.6669, 1.821, 3.19e-07, 0.0004654]
        }
    },
    'BIV_S1': {
        'ULV_spheric_asym_C4': {
            1: [16.46, 2.746, 1.365, 0.949, 0.35, 0.3118, 0.6541, 1.909, 0.0001544, 0.0005382]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.26, 2.946, 0.8713, 0.0916, 1.465, 1.049, 0.382, 0.3424, 0.7931, 1.941, 3.333e-07, 0.0003789],
            # 2: [16.27, 2.865, 0.7452, 0.3678, 1.366, 0.9596, 0.34, 0.3265, 0.6587, 1.929, 4.687e-07, 0.0004716],
            # 3: [16.34, 2.944, 0.732, 0.463, 1.271, 1.049, 0.34, 0.3135, 0.6394, 1.886, 4.841e-07, 0.0002168]
        }
    },
    'BIV_S2': {
        'ULV_spheric_asym_C4': {
            1: [9.984, 4.576, 1.277, 1.016, 0.3272, 0.3519, 0.6674, 1.906, 0.0003534, 0.0004885]
        },
        'ULV_ellipsoid_asym_C4': {
            1:  [10.18, 4.776, 0.9143, 0.2056, 1.884, 1.166, 0.3172, 0.3419, 0.8674, 1.858, 6.867e-07, 6.052e-05],
            # 2: [10.04, 4.772, 0.9709, 0.2317, 1.377, 1.108, 0.3204, 0.3421, 0.7087, 1.879, 6.977e-07, 0.0004477],
        }
    },
    'BIV_S3': {
        'ULV_spheric_asym_C4': {
            1: [10.25, 4.523, 1.236, 1.042, 0.3548, 0.3601, 0.7038, 1.888, 0.0003148, 0.0004641]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [10.26, 4.722, 0.9766, 0.1896, 1.294, 1.125, 0.3465, 0.3501, 0.7296, 1.878, 6.305e-07, 0.0004352]
        }
    },
    'BIV_S4': {
        'ULV_spheric_asym_C4': {
            1: [16.56, 3.14, 1.541, 0.8599, 0.3301, 0.3726, 0.705, 1.887, 0.0001312, 0.0004451]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.48, 3.33, 0.8856, 0.2918, 1.543, 0.9217, 0.3228, 0.3626, 0.7047, 1.893, 3.148e-07, 0.0004243]
        }
    },
    'BIV_S5': {
        'ULV_spheric_asym_C4': {
            1: [16.05, 3.551, 1.408, 0.9312, 0.3456, 0.3181, 0.6606, 1.901, 0.0001454, 0.0004653]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.01, 3.748, 0.9081, 0.2396, 1.508, 1.018, 0.3356, 0.3169, 0.6991, 1.904, 3.16e-07, 0.0004296],
            2: [16.25, 3.751, 1.14, 0.2063, 1.405, 0.982, 0.3356, 0.3081, 0.6468, 1.801, 2.593e-07, 0.0004446]
        }
    },
    'BIV_S6': {
        'ULV_spheric_asym_C4': {
            1: [16.23, 3.431, 1.547, 0.8343, 0.3279, 0.3607, 0.6795, 1.891, 0.0001441, 0.0004692]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.03, 3.626, 0.9201, 0.2254, 1.635, 0.9342, 0.323, 0.3509, 0.7203, 1.891, 3.143e-07, 0.0004281]
        }
    },
    'BIV_S7': {
        'ULV_spheric_asym_C4': {
            1: [16.08, 4.223, 1.627, 0.8058, 0.3221, 0.3626, 0.6807, 1.888, 0.0001421, 0.0004869]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.03, 4.255, 0.7812, 0.4446, 1.536, 0.8167, 0.3121, 0.3526, 0.6357, 1.861, 4.324e-07, 0.0004522],
            2: [16.27, 4.376, 0.9152, 0.338, 1.539, 0.8157, 0.3121, 0.3526, 0.6338, 1.859, 3.516e-07, 0.0003248]
        }
    },
    'BIV_S8': {
        "flat_asym_C4_q0p7": {
            1: [0.2442, 0.3632, 0.6252, 1, 1.684, 1.89, 0.04044, 0.0006435],
        },
        'ULV_spheric_asym_C4': {
            1: [16.15, 5.084, 1.641, 0.8074, 0.3176, 0.3193, 0.6389, 1.9, 0.0001657, 0.0006746]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.35, 5.284, 1.098, 0.1713, 1.605, 0.8125, 0.3115, 0.3098, 0.6131, 1.826, 3.133e-07, 0.0006509],
            # 2: [16.34, 5.264, 0.9524, 0.277, 1.617, 0.8386, 0.3114, 0.3099, 0.6227, 1.873, 3.687e-07, 0.0005159],
            # 3: [16.32, 5.281, 0.8406, 0.3059, 1.657, 0.8572, 0.31, 0.3097, 0.6366, 1.918, 4.142e-07, 0.0006252],
            4: [16.35, 5.284, 1.15, 0.1457, 1.636, 0.8238, 0.3076, 0.3093, 0.6191, 1.8, 2.949e-07, 0.000652],
        }
    },
    'BIV_S9': {
        "flat_asym_C4_q0p7": {
            1: [0.291, 0.4027, 0.7033, 1, 2.036, 1.857, 0.02832, 0.0006657]
        },
        'ULV_spheric_asym_C4': {
            1: [21.07, 7.182, 1.729, 0.8, 0.3169, 0.3059, 0.6517, 1.887, 8.572e-05, 0.000683]
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
            2: [17.48, 6.869, 1.703, 1.33, 0.1981, 0.4524, 0.814, 1.807, 0.0001049, 0.0008936]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [9.835, 6.916, 1.066, 0.02601, 0.8369, 1.826, 0.315, 0.2664, 0.6469, 1.842, 5.358e-07, 0.0006962],
            # 2: [9.777, 6.853, 1.014, 0.06389, 0.8536, 1.799, 0.3155, 0.2743, 0.6594, 1.871, 5.622e-07, 0.0007022],
        }
    },
    'BIV_S21': {
        'ULV_spheric_asym_C4': {
            1: [9.439, 4.549, 1.285, 1.029, 0.327, 0.3481, 0.6689, 1.909, 0.0004254, 0.0005276]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [9.371, 4.747, 0.9896, 0.1953, 1.347, 1.097, 0.317, 0.338, 0.6874, 1.889, 8.572e-07, 0.0005051],
            2: [9.618, 4.747, 1.136, 0.1462, 1.219, 1.129, 0.3172, 0.3381, 0.6633, 1.809, 7.41e-07, 0.0004558]
        }
    },
    'BIV_S22': {
        'ULV_spheric_asym_C4': {
            1: [16.12, 2.826, 1.511, 0.8386, 0.3237, 0.3622, 0.6723, 1.903, 0.0001828, 0.0005951]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.05, 3.025, 0.8926, 0.2745, 1.533, 0.9026, 0.3138, 0.3534, 0.6768, 1.909, 4.314e-07, 0.0005479],
            2: [16.32, 3.025, 1.11, 0.2161, 1.47, 0.8752, 0.3151, 0.3522, 0.6483, 1.816, 3.425e-07, 0.0005312]
        }
    },
    'BIV_S23': {
        'ULV_spheric_asym_C4': {
            1: [16.14, 2.666, 1.361, 0.9862, 0.3525, 0.2982, 0.6554, 1.907, 0.0001876, 0.0005509]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.97, 2.866, 0.9517, 0.2707, 1.38, 1.081, 0.3424, 0.2889, 0.6609, 1.885, 4.147e-07, 0.0005221]
        }
    },
    'BIV_S24': {
        'ULV_spheric_asym_C4': {
            1: [16.14, 2.714, 1.363, 0.9531, 0.3528, 0.3093, 0.6559, 1.904, 0.0002031, 0.0005877]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.95, 2.914, 0.9734, 0.1933, 1.399, 1.021, 0.3429, 0.2993, 0.6615, 1.898, 4.301e-07, 0.000578]
        }
    },
    'BIV_S25': {
        'ULV_spheric_asym_C4': {
            1: [15.93, 3.017, 1.378, 0.9368, 0.3497, 0.3141, 0.6549, 1.903, 0.0002164, 0.0006444]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.73, 2.817, 0.8714, 0.06079, 1.478, 1.037, 0.367, 0.324, 0.7541, 1.954, 4.747e-07, 0.0005007],
            # 2: [15.77, 3.213, 0.9435, 0.2109, 1.44, 1.011, 0.3397, 0.3078, 0.6749, 1.906, 4.64e-07, 0.0006092],
            # 3: [16.07, 3.216, 1.146, 0.2052, 1.338, 0.949, 0.34, 0.3127, 0.6291, 1.803, 3.961e-07, 0.0006117]
        }
    },
    'BIV_S26': {
        'ULV_spheric_asym_C4': {
            1: [9, 5.06, 1.221, 1.086, 0.336, 0.3382, 0.6656, 1.9, 0.0004797, 0.0006352]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [8.807, 5.248, 0.9859, 0.09175, 1.249, 1.137, 0.326, 0.3282, 0.6726, 1.899, 1.026e-06, 0.0005477]
        }
    },
    'BIV_S27': {
        'ULV_spheric_asym_C4': {
            1: [9, 5.376, 1.139, 1.186, 0.3437, 0.3312, 0.6719, 1.892, 0.0004574, 0.0006214]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [9.021, 5.541, 0.8988, 0.2963, 1.134, 1.171, 0.3341, 0.3257, 0.6508, 1.889, 1.111e-06, 0.0006048],
            2: [9.146, 5.575, 1.111, 0.1403, 1.118, 1.286, 0.3338, 0.3248, 0.6826, 1.803, 8.026e-07, 0.000538],
            3: [9.147, 5.548, 0.9825, 0.2466, 1.077, 1.159, 0.3345, 0.3264, 0.6303, 1.875, 1.002e-06, 0.0004942]
        }
    },
    'BIV_S28': {
        'ULV_spheric_asym_C4': {
            1: [15.42, 4.433, 1.558, 0.8492, 0.3361, 0.3338, 0.6703, 1.888, 0.0002257, 0.0008178]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [15.36, 4.631, 0.9491, 0.2631, 1.609, 0.9179, 0.3263, 0.324, 0.6809, 1.869, 4.88e-07, 0.0007793],
            2: [15.57, 4.632, 1.132, 0.1746, 1.519, 0.8601, 0.3305, 0.3242, 0.6442, 1.795, 4.163e-07, 0.0007935]
        }
    },
    'BIV_S29': {
        'ULV_spheric_asym_C4': {
            1: [16.83, 5.413, 1.714, 0.8, 0.3243, 0.3126, 0.6564, 1.888, 0.0001515, 0.0007714]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.98, 5.612, 1.137, 0.1337, 1.731, 0.8443, 0.3161, 0.3028, 0.6542, 1.79, 2.681e-07, 0.0007421],
            2: [16.86, 5.613, 0.8785, 0.292, 1.702, 0.8753, 0.329, 0.3026, 0.6707, 1.89, 3.57e-07, 0.0007345]
        }
    },
    'BIV_S30': {
        'ULV_spheric_asym_C4': {
            1: [16.8, 6.039, 1.861, 0.8, 0.3081, 0.2829, 0.6389, 1.891, 0.0001453, 0.000664]
        },
        'ULV_ellipsoid_asym_C4': {
            1: [16.99, 6.239, 1.168, 0.01873, 1.82, 0.8121, 0.2981, 0.2729, 0.6101, 1.791, 2.583e-07, 0.000682],
            # 2: [16.65, 6.074, 0.8248, 0.3252, 1.849, 0.8581, 0.3066, 0.2729, 0.6399, 1.894, 3.921e-07, 0.0006033]
        }
    }
}

EDP_lib = {
    # Each value =          [sample,                    rho_h1, rho_h2, sigma_h1, sigma_h2,  sigma_c,    z_h]
    "HA5_CA5_LV4_25uMaSyn": ["HA5_CA5_LV4_25uMaSyn",    1.4986, 1.7302,  0.37497,  0.20002,  0.75887, 1.8633],
    "BII_S9": [],
    "BIV_S8": [],
    "BIV_S9": [],
    "BIV_S10": [],
}