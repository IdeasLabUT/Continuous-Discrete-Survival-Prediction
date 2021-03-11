from data_downloader import download_file
from aaai_mtlr_func import run_experiment as run_mtlr
from aaai_coxph_func import run_experiment as run_coxph

from datasets import get_flchain, get_whas500, get_DBCD, get_NWTCO


print("Downloading DBCD...")
download_file("http://user.it.uu.se/~liuya610/data/bovelstaddata.mat", "./datasets/dbcd.mat")

print("Downloading NWTCO...")
download_file("https://r-data.pmagunia.com/system/files/datasets/dataset-71644.csv", "./datasets/nwtco.csv")

for dataset_ in [get_flchain, get_whas500, get_DBCD, get_NWTCO]:
    for num_bins_ in [2, 5, 10, 15, 20, 25]:
        run_mtlr(dataset_, num_bins_)

for dataset_ in [get_flchain, get_whas500, get_DBCD, get_NWTCO]:
    for num_bins_ in [0, 5, 10, 15, 20, 25]:
        run_coxph(dataset_, num_bins_)