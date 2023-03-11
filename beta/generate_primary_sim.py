# harness to run BogoEnv beta gym environment
# from "runenv.py"
# JMA 7 March 2023
import os, re, sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import gym

sys.path.append('./RL_offline/')
import envs_beta

EPISODE_LEN = 22
# HISTORY_FILE = "../patient_data_random_doseE6.csv"
CONST_ACTION = 0.9
NEIGHBORS = 2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = 'What the program does')

    parser.add_argument('simulation file',
                        default='output.parquet',
                        help='output file')   # positional argument
    parser.add_argument('-e', '--episodes',
                        help = 'episode maximum length',
                        default = 25,
                        type=int)            # option that takes a numeric value
    parser.add_argument('-c', '--cohorts',
                        help = 'cohorts for RCTs',
                        default = 1,
                        type=int)           
    parser.add_argument('-s', '--samples',
                        help = 'samples, i.e. patients per cohort',
                        default = 10,
                        type=int)            
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    # values can be found in a named tuple: args.filename, args.count, args.verbose
	
    
    # A column for each variable: reward, Drug-action, Infection, CumulativeDrug, Severity, reward 
    run_trajectory = np.empty((EPISODE_LEN, 5))

    bg_env = gym.make('BogoEnvBeta-Acc-v0', disable_env_checker=True)
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
        # a = bg_env.action_space.sample()
        a = CONST_ACTION
        if terminated or truncated:
            observation, info = bg_env.reset()
            last_episode = i
            break

    bg_env.close()

    lbls = ['Reward','Dose'] + list(observation.keys())
    run = pd.DataFrame(run_trajectory, columns=lbls).iloc[0:last_episode+1,:]
    
    # num_recovered = np.sum(patient_data.outcome == 'recover')
    # num_died = np.sum(patient_data.outcome == 'die')
    # print(f"{num_recovered} patients recovered and {num_died} died")
    # print(len(run))