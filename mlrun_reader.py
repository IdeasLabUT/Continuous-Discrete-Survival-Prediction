import os
import glob
import numpy as np
import ntpath
import yaml
import ntpath
ntpath.basename("a/b/c")

PATH = "./mlruns"
print(f"{str('Experiment ID').ljust(30, ' ')} {str('Experiment Name').ljust(90, ' ')} {str('Test C Index').ljust(20, ' ')} {str('Best Val C Index').ljust(20, ' ')} {str('Train C Index').ljust(20, ' ')} {str('ID').ljust(20, ' ')}")

exps = []
for exp_id in os.listdir(PATH):
    try:
        string_int = int(exp_id)
        exps.append(string_int)
    except ValueError:
        pass

for exp_id in sorted(exps):
    if os.path.isdir(f"{PATH}/{exp_id}"):
        with open(f"{PATH}/{exp_id}/meta.yaml", 'r') as file:
            try:
                meta_file = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        exp_name = meta_file["name"]
        best_params = {}
        best_test = 0
        best_train = 0
        best_val = 0
        best_id = ""
        num_failed = 0
        count = 0
        for file in glob.glob(f"{PATH}/{exp_id}/*"):
            path = os.path.join(file, "metrics/mean-val-C-index")
            if os.path.isfile(path):
                val_score = np.loadtxt(path, delimiter=" ")[1]
                if type(val_score) not in [float, np.float64]:
                    continue
                id_ = ntpath.basename(file)
                count += 1

                if val_score == 0:
                    num_failed += 1

                if val_score > best_val:
                    best_val = val_score
                    best_id = id_

                    path = os.path.join(file, "metrics/mean-train-C-index")
                    if os.path.isfile(path):
                        train_score = np.loadtxt(path, delimiter=" ")[1]
                        best_train = train_score

                    path = os.path.join(file, "metrics/mean-test-C-index")
                    if os.path.isfile(path):
                        test_score = np.loadtxt(path, delimiter=" ")[1]
                        best_test = test_score

                    for param in glob.glob(os.path.join(file, "params/*")):
                        if os.path.isfile(param) and not any(ext in param for ext in ["logger_config", "key"]):
                            param_val = np.loadtxt(param, delimiter=" ")
                            param_name = ntpath.basename(param)
                            best_params[param_name] = param_val

            else:
                num_failed += 1

        params_out = ""
        for key in best_params.keys():
            params_out += str(f" {key}: {best_params[key]}").ljust(35, ' ')

        print(f"{str(exp_id).ljust(30, ' ')} "
              f"{str(exp_name).ljust(90, ' ')} "
              f"{str(best_test).ljust(20, ' ')} "
              f"{str(best_val).ljust(20, ' ')} "
              f"{str(best_train).ljust(20, ' ')} "
              f"{str(best_id).ljust(20, ' ')} "
              f"{str(f' failed: {num_failed}').ljust(15, ' ')} "
              f"{str(f' total: {count}').ljust(15, ' ')} "
              f"{params_out}")