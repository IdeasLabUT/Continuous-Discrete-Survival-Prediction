# Continuous-Discrete-Survival-Prediction
This repository provides a complete code replication for the paper Empirical Comparison of Continuous and Discrete-time 
Representations for Survival Prediction (_Michael Sloma, Fayeq Syed, Mohammedreza Nemati and Kevin S. Xu_) presented at
AAAI Symposium 2021 Workshop on Survival Prediction: Algorithms, Challenges and Applications (SP-ACA).

## Setting Up The Environment
* We utilized [Anaconda](https://www.anaconda.com/) for our environment/package management but you may also use the PIP virtual environment.
* *[PySurvival](https://square.github.io/pysurvival/)* and *[Ray](https://docs.ray.io/en/master/tune/index.html)* do not play nicely with Windows, so we recommend using Linux (We used Ubuntu 18.04).
* We utilized Python 3.8.5

Once you have set up your Python environment you can install the requirements needed by navigating to this 
project's directory and running `pip install -r requirements.txt`

## Running The Code
To download all the data automatically and run both the CoxPH and MTLR models, all you need to do is run the `main.py` file.

### Downloading the Data
If all you want to do is download the data used in this paper that *is not scikit-survival data*, 
you can run the `data_downloader.py` file.

### Running the CoxPH Models
If you want to run the CoxPH models and have *optionally* downloaded the data, you can run the `aaai_coxph_func.py`. 
If you have not downloaded the data, it will do it for you.

### Running the MTLR Models
If you want to run the MTLR models and have *optionally* downloaded the data, you can run the `aaai_mtlr_func.py`. 
If you have not downloaded the data, it will do it for you.

## Accessing The Results
The easiest way to access the results is to use [MLFlow](https://mlflow.org/). After you have run 
the experiments you can run `mlflow ui` in this directory. We have noted that on the MTLR experiments with 5000 samples
that MLflow may run into a server error. We have included an optional secondary method for accessing the results in 
`mlrun_reader.py` which will "manually" read the results in the `./mlruns` directory for you.