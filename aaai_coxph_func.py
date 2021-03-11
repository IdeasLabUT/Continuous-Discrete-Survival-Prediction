import numpy as np

import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS

from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.logger import CSVLogger, JsonLogger, MLFLowLogger

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow.tracking import MlflowClient

from datasets import get_flchain, get_whas500, get_DBCD, get_NWTCO


def trainer(config, data_split=None, data=None, labels=None, cont_labels=None):
    print(f"Alpha: {config['alpha']}")
    alpha = config["alpha"]
    epochs = config["epochs"]


    mlflow.log_param("epochs", epochs)
    mlflow.log_param("alpha", alpha)
    try:
        train_scores = []
        val_scores = []
        test_scores = []

        for train_index, val_index, test_index in data_split:
            data_train, data_val, data_test = data[train_index], data[val_index], data[test_index]
            labels_train, labels_val, labels_test = labels[train_index], labels[val_index], labels[test_index]

            cont_labels_train, cont_labels_val, cont_labels_test = cont_labels[train_index], cont_labels[val_index], cont_labels[test_index]

            scaler = StandardScaler().fit(data_train)
            data_train = scaler.transform(data_train)
            data_val = scaler.transform(data_val)
            data_test = scaler.transform(data_test)

            lifetime_train = labels_train[:, 0]
            censor_train = labels_train[:, 1]

            cont_lifetime_train = cont_labels_train[:, 0]

            cont_lifetime_val = cont_labels_val[:, 0]
            censor_val = cont_labels_val[:, 1]

            cont_lifetime_test = cont_labels_test[:, 0]
            censor_test = cont_labels_test[:, 1]

            data_surv_train = np.zeros(censor_train.shape[0], dtype={'names': ('censor', 'time'),
                                                                     'formats': (bool, float)})
            data_surv_train["censor"] = censor_train
            data_surv_train["time"] = lifetime_train

            print("fitting...")
            model = CoxPHSurvivalAnalysis(alpha=alpha,
                                          ties="efron",
                                          n_iter=epochs,
                                          verbose=1).fit(data_train, data_surv_train)

            print("done fitting...")
            train_risk = model.predict(data_train)
            c_index_train = concordance_index_censored(censor_train.astype(bool), cont_lifetime_train, train_risk)[0]
            print('Train C-index: {:.6f}'.format(c_index_train))
            train_scores.append(c_index_train)

            val_risk = model.predict(data_val)
            c_index_val = concordance_index_censored(censor_val.astype(bool), cont_lifetime_val, val_risk)[0]
            print('Validation C-index: {:.6f}'.format(c_index_val))
            val_scores.append(c_index_val)

            test_risk = model.predict(data_test)
            c_index = concordance_index_censored(censor_test.astype(bool), cont_lifetime_test, test_risk)[0]
            print('Test C-index: {:.6f}'.format(c_index))
            test_scores.append(c_index)

        result_dict = {"mean-test-C-index": np.mean(test_scores),
                       "max-test-C-index": max(test_scores),
                       "min-test-C-index": min(test_scores),

                       "mean-val-C-index": np.mean(val_scores),
                       "max-val-C-index": max(val_scores),
                       "min-val-C-index": min(val_scores),

                       "mean-train-C-index": np.mean(train_scores),
                       "max-train-C-index": max(train_scores),
                       "min-train-C-index": min(train_scores)}

    except ValueError as e:
        print(e)
        result_dict = {"mean-test-C-index": 0,
                       "max-test-C-index": 0,
                       "min-test-C-index": 0,

                       "mean-val-C-index": 0,
                       "max-val-C-index": 0,
                       "min-val-C-index": 0,

                       "mean-train-C-index": 0,
                       "max-train-C-index": 0,
                       "min-train-C-index": 0}

    return tune.report(**result_dict)


def run_experiment(dataset, num_bins):
    data, labels, name = dataset()

    print(f"data nan: {np.isnan(data).any()}")
    print(f"data inf: {np.isinf(data).any()}")
    print(f"labels nan: {np.isnan(labels).any()}")
    print(f"labels inf: {np.isinf(labels).any()}")

    data, labels = shuffle(data, labels, random_state=0)
    cont_labels = labels.copy()

    def get_data_split(folds=None):
        if folds:
            kf = KFold(n_splits=folds, random_state=0)
            data_split = list(kf.split(data))
        else:
            data_split = [(range(0, int(round(data.shape[0] * 0.6))),
                           range(int(round(data.shape[0] * 0.6)), int(round(data.shape[0] * 0.8))),
                           range(int(round(data.shape[0] * 0.8)), data.shape[0])), ]

        return data_split

    def discritization(num_bins):
        bins = np.linspace(np.min(labels[:, 0]), np.max(labels[:, 0]), num=num_bins)
        bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        for bin_lower, bin_upper in bins:
            mid = (bin_lower + bin_upper) / 2
            loc = np.where((bin_lower <= labels[:, 0]) & (labels[:, 0] <= bin_upper))
            labels[loc, 0] = mid

    if num_bins != 0:
        discritization(num_bins=num_bins)
    data_split = get_data_split()

    epochs = CSH.UniformIntegerHyperparameter(name=f"epochs", lower=20, upper=1000, log=False)
    alphas = CSH.UniformFloatHyperparameter(name=f"alpha", lower=10000, upper=10000000.0, log=True)

    config_space = CS.ConfigurationSpace(seed=1234)

    config_space.add_hyperparameters([alphas, epochs])

    experiment_metrics = dict(metric="mean-val-C-index", mode="max")
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration", max_t=1, **experiment_metrics)
    bohb_search = TuneBOHB(
        config_space, max_concurrent=10, **experiment_metrics)

    NAME = f"{name}_coxph_num_bins_{num_bins}"
    client = MlflowClient("./mlruns")
    experiments = client.list_experiments()
    experiment_id = None
    for experiment in experiments:
        if experiment.name == NAME:
            experiment_id = experiment.experiment_id
    if not experiment_id:
        experiment_id = client.create_experiment(NAME)

    analysis = tune.run(
        tune.with_parameters(trainer, data=data, labels=labels, cont_labels=cont_labels, data_split=data_split),
        name=NAME,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=100,
        resources_per_trial={"cpu": 1},
        loggers=[CSVLogger, JsonLogger, MLFLowLogger],
        config={
            "logger_config": {
                "mlflow_experiment_id": experiment_id
            },
        },
        local_dir="./tune-results",
        )


if __name__ == "__main__":
    for dataset_ in [get_flchain, get_whas500, get_DBCD, get_NWTCO]:
        for num_bins_ in [0, 5, 10, 15, 20, 25]:
            run_experiment(dataset_, num_bins_)
