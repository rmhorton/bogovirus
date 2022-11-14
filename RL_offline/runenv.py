# harness to run BogoEnv gym environment
# JMA 27 Sept 2022
import os, re, sys
import pandas as pd
import numpy as np
import gym

sys.path.append('/home/azureuser/cloudfiles/code/Users/joagosta/bogo/')
import envs
# See https://medium.com/swlh/how-to-setup-mlflow-on-azure-5ba67c178e7d
import mlflow

# If azureml-core and azureml-mlflow have been loaded 
# the setting the tracking uri is done by the mlflow package
# for you so this is not needed:
# from azureml.core import Workspace
# workspace = Workspace.from_config()
# mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
# The Tracking Server will have a url like "azureml://eastus2.api.azureml.ms/mlflow..."

EPISODE_LEN = 22
# HISTORY_FILE = "../patient_data_random_doseE6.csv"
CONST_ACTION = 0.9
NEIGHBORS = 2

# Also by running this on a compute context, the experiment contents will be displayable
# in the Azure studio jobs tab.  An Azure ML job corresponds to an MLflow experiment. 
# (Is there a way to run the mlflow ui for an Azure ML server? )
mlflow.set_experiment("BogoEnv-v0-tst2")

# A column for each variable: reward, Drug-action, Infection, CumulativeDrug, Severity, reward 
run_trajectory = np.empty((EPISODE_LEN, 5))

mlflow_run = mlflow.start_run()
bg_env = gym.make('BogoEnv-v0', disable_env_checker=True)
observation, info = bg_env.reset()
# Initialize
# a = bg_env.action_space.sample()
a = CONST_ACTION
k = NEIGHBORS
last_episode = EPISODE_LEN
run_trajectory[0] = [0, a] + list(observation.values())

for i in range(1, EPISODE_LEN):
    observation, reward, terminated, truncated, info = bg_env.step([k,a])
    run_trajectory[i] = [reward, a] + list(observation.values())
    mlflow.log_metrics(observation)
    # a = bg_env.action_space.sample()
    a = CONST_ACTION
    if terminated or truncated:
        observation, info = bg_env.reset()
        last_episode = i
        break

bg_env.close()

lbls = ['Reward','Dose'] + list(observation.keys())
run = pd.DataFrame(run_trajectory, columns=lbls).iloc[0:last_episode+1,:]
mlflow.end_run()
print(run)