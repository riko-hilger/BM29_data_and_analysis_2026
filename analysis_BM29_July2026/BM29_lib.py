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

# def blockpath(block):
#     # bpath = Path("C:/Users/rikoh/Documents/Physik Göttingen&Grenoble 2022-2026/8. SoSe 26/SHK Synaptische Proteine, Lipidmembranen SoSe 2026/Code/BM29_data_and_analysis/data_BM29_July2026") / block
#     bpath = Path("C:/Users/riko05/Documents/8. SoSe 26/Code/BM29_data_and_analysis_2026/data_BM29_July2026") / block
#     # bpath = Path("/home/AG_Salditt/Projekte_Membranbiophysik/PSCM_Riko/Code and data/BM29_data_and_analysis_2026/data_BM29_July2026") / block
#     return bpath

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
}