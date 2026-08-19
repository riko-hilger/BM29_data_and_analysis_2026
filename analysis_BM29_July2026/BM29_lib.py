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