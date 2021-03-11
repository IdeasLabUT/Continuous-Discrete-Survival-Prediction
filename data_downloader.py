import requests
import os
import scipy.io
import pandas as pd
import numpy as np


def download_file(url, save_filepath=None):
    if not os.path.isfile(save_filepath):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


def load_DBCD(dbcd_filepath):
    x, y = None, None
    if os.path.isfile(dbcd_filepath):
        mat = scipy.io.loadmat(dbcd_filepath)
        y = {"survival_censoring_days": mat["Y"].reshape((-1,)),
             "status": mat["delta"].reshape((-1,))}

        x = mat["X"]
        y = pd.DataFrame(y).to_numpy()

    return x, y, "DBCD"


def load_NWTCO(nwtco_filepath):
    # age,instit_1,instit_2,histol_1,histol_2,stage_1,stage_2,stage_3,stage_4,study_3,study_4,in.subcohort_False,in.subcohort_True
    nwtco_data = pd.read_csv(nwtco_filepath)
    y = {"survival_censoring_days": nwtco_data["edrel"].to_numpy(),
         "status": nwtco_data["rel"].to_numpy()}

    nwtco_data.drop(["seqno", "rel", "edrel"], axis=1, inplace=True)

    for col in ["instit", "histol", "stage", "study"]:
        unique_vals = sorted(nwtco_data[col].unique())
        for val in unique_vals:
            nwtco_data[f"{col}_{val}"] = np.where(nwtco_data[col] == val, 1, 0)
        nwtco_data.drop([col], axis=1, inplace=True)

    nwtco_data["in.subcohort_False"] = np.where(nwtco_data["in.subcohort"] == False, 1, 0)
    nwtco_data["in.subcohort_True"] = np.where(nwtco_data["in.subcohort"] == True, 1, 0)
    nwtco_data.drop(["in.subcohort"], axis=1, inplace=True)

    x = nwtco_data.to_numpy()
    y = pd.DataFrame(y).to_numpy()

    return x, y, "NWTCO"


if __name__ == "__main__":
    print("Downloading DBCD...")
    download_file("http://user.it.uu.se/~liuya610/data/bovelstaddata.mat", "./datasets/dbcd.mat")
    x, y, name = load_DBCD("./datasets/dbcd.mat")

    print("Downloading NWTCO...")
    download_file("https://r-data.pmagunia.com/system/files/datasets/dataset-71644.csv", "./datasets/nwtco.csv")
    x, y, name = load_NWTCO("./datasets/nwtco.csv")

