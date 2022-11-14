README: RL offline simulation

files:
------

iter_runenv.py                 - Run the offline "Bogovirus" environment and iterate over a range of drug dosages 

env/                           - OpenGym module for the offline simulation environment
    __init__.py                - module file
    bogo_world.py              - BogoEnv Class

patient_data_random_doseE6.csv - 1E6 simulation records. 

README.txt                     - This file


install:
--------

    To install on an Azure ML compute instance, the package install process is fragile, and conda has problems. I'm not able to build a conda env from scratch, by installing the required azureml-core and azureml-mlflow on top of python 3.9 in the "base" env. Neither does a conda install of gym from sourceforce on top of the azureml_py38 env work -- it installs obsolete gym version 0.21. 

Apparently install in azureml_py38  from the github.com/openai/gym works.  
    1. Clone the gygm repo,  cd into it. 
    2. Run $ python setup.py install
The env azureml_py38 already has the required azureml packages, mlflow and pandas installed. 

