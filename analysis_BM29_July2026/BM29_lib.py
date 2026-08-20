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
    plt.legend()
    return

def improved_Poisson_error_low_high(q, Ierr_Poisson):
    mask = (q <= 0.1) & (q >= 4.1)
    Ierr_Poisson_improved = Ierr_Poisson.copy()
    Ierr_Poisson_improved[mask] = Ierr_Poisson[mask] * 2
    return Ierr_Poisson_improved

def op_error_classic(q, I):
    mask = (q <= 0.1) & (q >= 4.1)
    Ierr_op = np.abs(I * 1/30)
    Ierr_op[mask] = np.abs(I[mask] * 1/15) 
    return Ierr_op

def op_error_focus_minimum_high(q, Ierr_Poisson):
    mask1 = (q <= 0.1)
    Ierr_op = Ierr_Poisson.copy()
    Ierr_op[mask1] = Ierr_op[mask1] * 2
    mask2 = ((q > 0.15) & (q < 0.4)) | (q >= 3)
    Ierr_op[mask2] = Ierr_op[mask2] * 0.1
    return Ierr_op

def op_error_focus_lowq_and_minimum(q, Ierr_Poisson):
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

    "BVIII_S1": "BVIII_S1_15mM_LV4_6uM_aSynWT",
    "BVIII_S2": "BVIII_S2_15mM_LV4_9uM_aSynWT",
    "BVIII_S3": "BVIII_S3_15mM_LV4_15uM_aSynWT",
    "BVIII_S4": "BVIII_S4_15mM_LV4_25uM_aSynWT",
    "BVIII_S5": "BVIII_S5_15mM_LV4_50uM_aSynWT",
    "BVIII_S6": "BVIII_S6_15mM_LV4_75uM_aSynWT",
    "BVIII_S7": "BVIII_S7_15mM_LV4_100uM_aSynWT",
    "BVIII_S8": "BVIII_S8_3mM_LV4_100uM_aSynWT",
    "BVIII_S9": "BVIII_S9_5mM_LV4_150uM_aSynWT",
    "BVIII_S11": "BVIII_S11_15mM_LV4_6uM_aSynG14R",
    "BVIII_S12": "BVIII_S12_15mM_LV4_9uM_aSynG14R",
    "BVIII_S13": "BVIII_S13_15mM_LV4_15uM_aSynG14R",
    "BVIII_S14": "BVIII_S14_15mM_LV4_25uM_aSynG14R",
    "BVIII_S15": "BVIII_S15_15mM_LV4_50uM_aSynG14R",
    "BVIII_S16": "BVIII_S16_15mM_LV4_75uM_aSynG14R",
    "BVIII_S17": "BVIII_S17_15mM_LV4_100uM_aSynG14R",
    "BVIII_S18": "BVIII_S18_3mM_LV4_100uM_aSynG14R",
    "BVIII_S19": "BVIII_S19_5mM_LV4_150uM_aSynG14R",
    "BVIII_S20": "BVIII_S20_15mM_LV4_6uM_aSynWT",
    "BVIII_S21": "BVIII_S21_15mM_LV4_9uM_aSynWT",
    "BVIII_S22": "BVIII_S22_15mM_LV4_15uM_aSynWT",
    "BVIII_S23": "BVIII_S23_15mM_LV4_25uM_aSynWT",
    "BVIII_S24": "BVIII_S24_15mM_LV4_50uM_aSynWT",
    "BVIII_S25": "BVIII_S25_15mM_LV4_75uM_aSynWT",
    "BVIII_S26": "BVIII_S26_15mM_LV4_100uM_aSynWT",
    "BVIII_S27": "BVIII_S27_3mM_LV4_100uM_aSynWT",
    "BVIII_S28": "BVIII_S28_5mM_LV4_150uM_aSynWT",
    "BVIII_S29": "BVIII_S29_5mM_LV4_250uM_aSynWT",

    "BIX_S1": "BIX_S1_15mM_POPG",
    "BIX_S2": "BIX_S2_15mM_DOPC_GM1_80_20",
    "BIX_S3": "BIX_S3_15mM_LV4_1uM_aSynG14R_1_25",
    "BIX_S4": "BIX_S4_15mM_LV4_2uM_aSynG14R_1_25",
    "BIX_S5": "BIX_S5_15mM_LV4_3uM_aSynG14R_1_25",
    "BIX_S6": "BIX_S6_15mM_LV4_6uM_aSynG14R_1_25",
    "BIX_S7": "BIX_S7_15mM_LV4_9uM_aSynG14R_1_25",
    "BIX_S8": "BIX_S8_15mM_LV4_15uM_aSynG14R_1_25",
    "BIX_S9": "BIX_S9_15mM_LV4_25uM_aSynG14R_1_25",
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
    "BII_S9": {
        "ULV_spheric_asym_C4": {
            1: [16.09, 2.936, 1.383, 0.9795, 0.3527, 0.3049, 0.6648, 1.907, 0.0001727, 0.0005608]
        }
    },
    "BIV_S1": {
        "ULV_spheric_asym_C4": {
            1: [16.46, 2.746, 1.365, 0.949, 0.35, 0.3118, 0.6541, 1.909, 0.0001544, 0.0005382]
        }
    },
    "BIV_S2": {
        "ULV_spheric_asym_C4": {
            1: [9.985, 4.576, 1.277, 1.016, 0.3272, 0.352, 0.6674, 1.906, 0.0003534, 0.0004885]
        }
    },
    "BIV_S3": {
        "ULV_spheric_asym_C4": {
            1: [10.25, 4.523, 1.236, 1.042, 0.3548, 0.3601, 0.7038, 1.888, 0.0003148, 0.0004641]
        }
    },
    "BIV_S4": {
        "ULV_spheric_asym_C4": {
            1: [16.56, 3.14, 1.541, 0.8599, 0.3301, 0.3726, 0.705, 1.887, 0.0001312, 0.0004451]
        }
    },
    "BIV_S5": {
        "ULV_spheric_asym_C4": {
            1: [16.05, 3.551, 1.408, 0.9312, 0.3456, 0.3182, 0.6606, 1.901, 0.0001454, 0.0004653]
        }
    },
    "BIV_S6": {
        "ULV_spheric_asym_C4": {
            1: [16.23, 3.431, 1.547, 0.8343, 0.3279, 0.3607, 0.6795, 1.891, 0.0001441, 0.0004692]
        }
    },
    "BIV_S7": {
        "ULV_spheric_asym_C4": {
            1: [16.08, 4.223, 1.627, 0.8058, 0.3221, 0.3626, 0.6807, 1.888, 0.0001421, 0.0004869]
        }
    },
    "BIV_S8": {
        "ULV_spheric_asym_C4": {
            1: [16.15, 5.084, 1.641, 0.8074, 0.3176, 0.3193, 0.6389, 1.9, 0.0001657, 0.0006746]
        }
    },
    "BIV_S9": {
        "ULV_spheric_asym_C4": {
            1: [21.07, 7.182, 1.729, 0.8, 0.3169, 0.3059, 0.6517, 1.887, 8.572e-05, 0.000683]
        }
    },
    "BIV_S10": {
        "ULV_spheric_asym_C4": {
            1: [9.793, 6.719, 0.8183, 1.843, 0.3249, 0.2602, 0.6434, 1.882, 0.0002879, 0.0007089]
        }
    },
    "BIV_S21": {
        "ULV_spheric_asym_C4": {
            1: [9.439, 4.549, 1.285, 1.029, 0.327, 0.348, 0.6689, 1.909, 0.0004254, 0.0005276]
        }
    },
    "BIV_S22": {
        "ULV_spheric_asym_C4": {
            1: [16.12, 2.826, 1.511, 0.8386, 0.3237, 0.3622, 0.6723, 1.903, 0.0001828, 0.0005951]
        }
    },
    "BIV_S23": {
        "ULV_spheric_asym_C4": {
            1: [16.14, 2.666, 1.361, 0.986, 0.3524, 0.2983, 0.6554, 1.907, 0.0001876, 0.0005509]
        }
    },
    "BIV_S24": {
        "ULV_spheric_asym_C4": {
            1: [16.14, 2.714, 1.363, 0.9531, 0.3528, 0.3093, 0.6559, 1.904, 0.0002031, 0.0005877]
        }
    },
    "BIV_S25": {
        "ULV_spheric_asym_C4": {
            1: [15.93, 3.017, 1.378, 0.9368, 0.3497, 0.3141, 0.6549, 1.903, 0.0002164, 0.0006444]
        }
    },
    "BIV_S26": {
        "ULV_spheric_asym_C4": {
            1: [9, 5.06, 1.221, 1.086, 0.336, 0.3382, 0.6656, 1.9, 0.0004797, 0.0006352]
        }
    },
    "BIV_S27": {
        "ULV_spheric_asym_C4": {
            1: [9, 5.375, 1.139, 1.186, 0.3437, 0.3312, 0.6719, 1.892, 0.0004574, 0.0006214]
        }
    },
    "BIV_S28": {
        "ULV_spheric_asym_C4": {
            1: [15.42, 4.433, 1.558, 0.8492, 0.3361, 0.3338, 0.6703, 1.888, 0.0002257, 0.0008178]
        }
    },
    "BIV_S29": {
        "ULV_spheric_asym_C4": {
            1: [16.83, 5.413, 1.714, 0.8, 0.3243, 0.3126, 0.6564, 1.888, 0.0001515, 0.0007714]
        }
    },
    "BIV_S30": {
        "ULV_spheric_asym_C4": {
            1: [16.8, 6.039, 1.861, 0.8, 0.3081, 0.2829, 0.6389, 1.891, 0.0001453, 0.000664]
        }
    }
}