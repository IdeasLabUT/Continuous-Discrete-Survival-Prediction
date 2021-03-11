import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.logger import CSVLogger, JsonLogger, MLFLowLogger

from pysurvival.models.multi_task import LinearMultiTaskModel

from sksurv.metrics import concordance_index_censored

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow.tracking import MlflowClient

from datasets import get_flchain, get_whas500, get_DBCD, get_NWTCO


def trainer(config, num_bins=None, data_split=None, data=None, labels=None):
    epochs = config["epochs"]
    l2_reg = config["l2_reg"]
    l2_smooth = config["l2_smooth"]
    lr = config["lr"]

    mlflow.log_param("bins", num_bins)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("l2_reg", l2_reg)
    mlflow.log_param("l2_smooth", l2_smooth)
    mlflow.log_param("lr", lr)

    # run training code
    n_mtlr = LinearMultiTaskModel(bins=num_bins, auto_scaler=False)

    try:
        train_scores = []
        val_scores = []
        test_scores = []

        for train_index, val_index, test_index in data_split:
            data_train, data_val, data_test = data[train_index], data[val_index], data[test_index]
            labels_train, labels_val, labels_test = labels[train_index], labels[val_index], labels[test_index]

            scaler = StandardScaler().fit(data_train)
            data_train = scaler.transform(data_train)
            data_val = scaler.transform(data_val)
            data_test = scaler.transform(data_test)

            lifetime_train = labels_train[:, 0]
            censor_train = labels_train[:, 1]

            lifetime_val = labels_val[:, 0]
            censor_val = labels_val[:, 1]

            lifetime_test = labels_test[:, 0]
            censor_test = labels_test[:, 1]

            n_mtlr.fit(data_train, lifetime_train, censor_train, lr=lr, init_method='xav_uniform',
                       l2_reg=l2_reg, l2_smooth=l2_smooth, num_epochs=epochs, extra_pct_time=0.0)

            if len(n_mtlr.loss_values) == epochs:
                print("Computing C-index")
                train_risk = n_mtlr.predict_risk(data_train)
                c_index_train = concordance_index_censored(censor_train.astype(bool), lifetime_train, train_risk)[0]
                print('Train C-index: {:.6f}'.format(c_index_train))
                train_scores.append(c_index_train)

                val_risk = n_mtlr.predict_risk(data_val)
                c_index_val = concordance_index_censored(censor_val.astype(bool), lifetime_val, val_risk)[0]
                print('Validation C-index: {:.6f}'.format(c_index_val))
                val_scores.append(c_index_val)

                test_risk = n_mtlr.predict_risk(data_test)
                c_index = concordance_index_censored(censor_test.astype(bool), lifetime_test, test_risk)[0]
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
    print(f"labels nan: {np.isnan(labels).any()}")

    data, labels = shuffle(data, labels, random_state=0)

    def get_data_split(folds=None):
        if folds:
            kf = KFold(n_splits=folds, random_state=0)
            data_split = list(kf.split(data))
        else:
            data_split = [(range(0, int(round(data.shape[0] * 0.6))),
                           range(int(round(data.shape[0] * 0.6)), int(round(data.shape[0] * 0.8))),
                           range(int(round(data.shape[0] * 0.8)), data.shape[0])), ]

        return data_split

    data_split = get_data_split()

    epochs = CSH.UniformIntegerHyperparameter(name=f"epochs", lower=20, upper=1000, log=False)
    l2_reg = CSH.UniformFloatHyperparameter(name=f"l2_reg", lower=1e-4, upper=1e-1, log=True)
    l2_smooth = CSH.UniformFloatHyperparameter(name=f"l2_smooth", lower=1e-4, upper=1e-1, log=True)
    lr = CSH.UniformFloatHyperparameter(name=f"lr", lower=1e-6, upper=1e-4, log=True)

    config_space = CS.ConfigurationSpace(seed=1234)

    config_space.add_hyperparameters([l2_reg, epochs, l2_smooth, lr])

    experiment_metrics = dict(metric="mean-test-C-index", mode="max")
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration", max_t=1, **experiment_metrics)
    bohb_search = TuneBOHB(
        config_space, max_concurrent=200, **experiment_metrics)

    NAME = f"{name}_mtlr_num_bins_{num_bins}"
    client = MlflowClient("./mlruns")
    experiments = client.list_experiments()
    experiment_id = None
    for experiment in experiments:
        if experiment.name == NAME:
            experiment_id = experiment.experiment_id
    if not experiment_id:
        experiment_id = client.create_experiment(NAME)

    analysis = tune.run(tune.with_parameters(trainer, num_bins=num_bins, data=data, labels=labels, data_split=data_split),
                        name=NAME,
                        scheduler=bohb_hyperband,
                        search_alg=bohb_search,
                        num_samples=5000,
                        resources_per_trial={"cpu": 2},
                        loggers=[CSVLogger, JsonLogger, MLFLowLogger],
                        config={
                            "logger_config": {
                                "mlflow_experiment_id": experiment_id
                            },
                        },
                        local_dir="./tune-results"
                        )


if __name__ == "__main__":
    for dataset_ in [get_flchain, get_whas500, get_DBCD, get_NWTCO]:
        for num_bins_ in [2, 5, 10, 15, 20, 25]:
            run_experiment(dataset_, num_bins_)
