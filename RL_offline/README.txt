README: RL offline  and online simulation

files:

iter_runenv.py                 - Run the offline "Bogovirus" environment and iterate over a range of drug doses
iter_run_accurate.py           - Run the on-line simulation of the environment, over the range of constant drug doses
plot_dose_response.ipynb       - Plot dose-response from trajectory files, for constant dose policies
reduce_trajectory.py           - Extract survival data from a set of trajectory files
run_env_accurate.py            - Run the simulation on-line environment with constant policies
runenv.py                      - Run one episode, to test offline env
simulation.py                  - The online simulation 
transition.py                  - Compute a discrete  probability transition matrix (this wasn't used).

envs/                          - OpenGym module for the offline simulation environment
    __init__.py                - module file
    BogoOfflineEnv.py              - BogoEnv Class

envs_accurate/                 - OpenGym module wrapping the simulation code
     __init__.py
     bogo_accurate.py

README.txt                     - This file

install on Azure:
----------------

    To install on an Azure ML compute instance, the package install process is fragile, and conda has problems. 
    I'm not able to build a conda env from scratch, by installing the required azureml-core and azureml-mlflow 
    on top of python 3.9 in the "base" env. Neither does a conda install of gym from sourceforce on top of 
    the azureml_py38 env work -- it installs obsolete gym version 0.21. 

Apparently install in azureml_py38  from the github.com/openai/gym works.  
    1. Clone the gygm repo,  cd into it. 
    2. Run $ python setup.py install
The env azureml_py38 already has the required azureml packages, mlflow and pandas installed. 

