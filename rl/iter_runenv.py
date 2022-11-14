# harness to run BogoEnv gym environment
# JMA 27 Sept 2022
import os, re, sys
import time
import pandas as pd
import numpy as np
import gym

sys.path.append('/home/azureuser/cloudfiles/code/Users/joagosta/bogo/')
import envs
import mlflow

EPISODE_LEN = 25
HISTORY_FILE = "../patient_data_random_doseE6.csv"   # A million simulation records
REPS = 200     # Number of episodes to average 
# N_NEIGHBORS = 5   # k for nearest neighbors search


def one_run(the_k, the_dose):
    'Iterate over the range of doses, running "reps" replications for one episode'
    # A column for each variable: reward, Drug-action, Infection, CumulativeDrug, Severity 
    run_trajectory = np.empty((EPISODE_LEN, 5))

    last_episode = EPISODE_LEN
    # Starting stage
    observation, info = bg_env.reset()
    run_trajectory[0] = [0, the_dose] + list(observation.values())

    # Transition until termination
    for i in range(1, EPISODE_LEN):
        observation, reward, terminated, truncated, info = bg_env.step([the_k, the_dose])
        run_trajectory[i] = [reward, the_dose] + list(observation.values())
        if terminated or truncated:
            last_episode = i
            observation.update({'Reward': reward, 'Stage': last_episode})
            mlflow.log_metrics(observation)
            observation, info = bg_env.reset()
            break
    #  Return the state and action for each stage in the episode. 
    return run_trajectory[1:(last_episode+1),:]

#### MAIN ########################################################################

mlflow.set_experiment("BogoEnv-v0-iter")

mlflow_run = mlflow.start_run()
bg_env = gym.make('BogoEnv-v0', disable_env_checker=True)
mlflow.log_param('Reps', REPS)
start_time = time.time()
output_dir = 'output'+str(round(start_time) - 1664000000)
os.makedirs(output_dir)

run_stats = dict()
for neighbors in (2,3,5,8):
    for dose in range(0, 15):
        one_run_stats = []   # Accumulate survival values 

        dose = dose/10.0

        dose_log = None     # Accumulate episode state action table. 
        for i in range(REPS):
            obs = one_run(neighbors, dose)
            # save the episode trajectory
            lbls = ['Reward','Dose', "Infection", "Severity", "CumulativeDrug"] 
            if dose_log is None:
                dose_log = pd.DataFrame(obs, columns=lbls)
            else:
                dose_log = pd.concat([dose_log, pd.DataFrame(obs, columns=lbls)], axis=0)
            fin_r = obs[obs.shape[0]-1, 0]        #the last reward tells recover or die
            live_or_die = int(fin_r/abs(fin_r) if fin_r != 0 else 0)
            one_run_stats.append(live_or_die)
            print('/',end='') # Keep from getting bored. 
        run_stats.update({str(round(dose,2)): one_run_stats})
        dose_log.to_csv(f'{output_dir}/trajectory_{neighbors}_{dose}.csv')  # Save the trajectories for each episode, for each dose level. 
        print(neighbors, dose, round(time.time() - start_time))

df_run_stats = pd.DataFrame(run_stats)
df_run_stats = df_run_stats.applymap(lambda k: 1 if k==1 else 0).mean(axis=0)
print('\nstats:\n',df_run_stats)   # Survival fraction 
dfrs = df_run_stats.to_dict()
tk =run_stats.keys()
rs = {k: dfrs[float(k)] for k in tk}
mlflow.log_metrics(rs)
df_run_stats.to_csv('run_stats100.csv')
mlflow.log_artifact('run_stats100.csv')

bg_env.close()
mlflow.end_run()
