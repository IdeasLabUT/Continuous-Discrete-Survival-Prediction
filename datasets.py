import sksurv.datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from data_downloader import load_DBCD, load_NWTCO


def get_flchain():
    data, labels = sksurv.datasets.load_flchain()
    gender = {'M': 0, 'F': 1}
    data.sex = [gender[item] for item in data.sex]

    y_n = {'no': 0, 'yes': 1}
    data.mgus = [y_n[item] for item in data.mgus]

    def to_hot(col_name):
        chapter = data[col_name].cat.codes.values.copy().reshape((-1, 1))
        chapter[np.where(chapter == -1)] = np.max(chapter) + 1
        encoded = OneHotEncoder(categories='auto').fit_transform(chapter)
        for i in range(encoded.shape[1]):
            data[f"{col_name}_{i}"] = encoded[:, i].toarray()
        data.drop(col_name, inplace=True, axis=1)

    to_hot("chapter")
    to_hot("flc.grp")
    to_hot("sample.yr")

    data = data.values
    data = np.nan_to_num(data)

    death = {False: 0, True: 1}
    labels_d = np.array([death[item] for item in labels["death"]])
    labels = np.concatenate([labels["futime"].reshape((-1, 1)), labels_d.reshape((-1, 1))], axis=1)

    return data, labels, "flchain"


def get_whas500():
    data, labels = sksurv.datasets.load_whas500()

    data = data.to_numpy().astype(float)

    death = {False: 0, True: 1}
    labels_d = np.array([death[item] for item in labels["fstat"]])
    labels = np.concatenate([labels["lenfol"].reshape((-1, 1)), labels_d.reshape((-1, 1))], axis=1)
    return data, labels, "whas500"


def get_DBCD():
    data, labels, name = load_DBCD("./datasets/dbcd.mat")

    return data, labels, name


def get_NWTCO():
    data, labels, name = load_NWTCO("./datasets/nwtco.csv")

    return data, labels, "NWTCO"


if __name__ == "__main__":
    for dataset in [get_flchain, get_whas500, get_DBCD, get_NWTCO]:
        data, labels, name = dataset()
        print(f"Name: {name} | Shape: {data.shape} | Censored: {data.shape[0]-np.sum(labels[:, 1])} | %: {1-np.sum(labels[:, 1])/data.shape[0]}")