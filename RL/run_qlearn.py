# harness to run qlearning (off policy TD) in BogoEnv gym online environment
# JMA 1 Nov 2022
# $ conda activate gymrl

import os, re, sys
import time
import pandas as pd
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import gym
import faiss
# from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parent))
import envs_accurate
import mlflow

RUN_FLOW = False  # Don't create ml flow records while debugging
EPISODE_LEN = 24  # Maximum number of stages 
REPS = 100        # Number of episodes to run
unique_dt = time.strftime( 'T%j-%H-%M-%S')  # A string, unique to the second

### Data structures
#  observation - the_state: dict
#  {'Infection': , 'Severity': , 'CumulativeDrug': }
#  trajectory - full history: list
#  [reward, action].extend(observation.values())
#  policy model - linear state weights: np.array(3, float)
#  np.array(i, s, c)

SEED = 0
# Const  policy runs - used to init the q function 
os.chdir('/home/azureuser/cloudfiles/code/Users/joagosta/')
HISTORY_FILE = "patient_data_random_doseE6.csv"
NROWS = 1000     # subsample the data



def lin_policy(the_state, the_model):
    'A policy that is linear in the state variables, with a weight learnt for each variable'
    vars = list(the_state.values())
    # Update the weights
    
    # Apply the weights
    action = None
    return action

def init_q():
    'Load a previous history as the starting point for the Q function.'
    # Pick one out of 1000
    my_rng = default_rng(seed=SEED)
    selector = lambda x: my_rng.random(size=1)[0] <= 0.001
    #selections = np.unif(history_file_len) < NROWS/ history_file_len
    q_history = pd.read_csv(HISTORY_FILE, header=0, index_col=0, skiprows=selector)
    # Select a "rich" subset of the history. 
    # Use the runs at const policy to bootstrap the q function.
    predictor = q_history[['infection_prev', 'severity', 'cum_drug_prev', 'drug']].values.astype('float32')
    knn_index = faiss.IndexFlatL2(predictor.shape[1])   # build the index
    knn_index.add(np.ascontiguousarray(predictor).astype('float32'))
    if not knn_index.is_trained:
        print('ERROR, knn index training failed')
    return knn_index

def q_approximation(s, a):
    ''
    # Use the set of points discovered in rollouts to predict value 
    # By nearest neighbor. 

def q_update(q_prev, the_alpha, greediness, the_env):
    ''
    # The greedy choice of action
    exploit = np.argmax(q[state])
    # The exploratory random choice is a randomization of the exploit action.
    explore= np.choose(q[state])
    the_dose = (1 - greediness) * explore + greediness * exploit
    #  Check the dose limits
    # UPDate: gamma discount == 1 for finite horizon RL
    observation, reward, terminated, truncated, info = the_env.step(the_dose) 
    q_now = q_prev + the_alpha * (reward + q - q_prev)
    
    
def one_episode(the_dose, the_env):
    'Iterate over the range of doses, running "reps" replications for one episode'
    # A column for each variable: reward, Drug-action, Infection, CumulativeDrug, Severity 

    run_trajectory = np.empty((EPISODE_LEN, 5))
    last_episode = EPISODE_LEN
    # Starting stage
    observation, info = the_env.reset()
    run_trajectory[0] = [0, the_dose] + list(observation.values())

    # Transition until episode termination
    for i in range(1, EPISODE_LEN):
        observation, reward, terminated, truncated, info = the_env.step(the_dose)
        run_trajectory[i] = [reward, the_dose] + list(observation.values())
        if terminated or truncated:
            last_episode = i
            observation.update({'Reward': reward, 'Stage': last_episode})
            if RUN_FLOW: mlflow.log_metrics(observation)
            observation, info = the_env.reset()
            break
    #  Return the state and action for each stage in the episode.
    return run_trajectory[1:(last_episode+1),:]

def run_reps(the_dose, run_stats={}):
    import envs_accurate
    bg_env = gym.make('BogoEnv-Acc-v0', disable_env_checker=True)
    run_reps_stats = []   # Accumulate outcomes for each rep 

    dose_log = None     # Accumulate episode state action table. 
    for i in range(REPS):
        episode_trajectory = one_episode(the_dose, bg_env)
        # save the episode trajectory
        lbls = ['Reward','Dose', "Infection", "Severity", "CumulativeDrug"] 
        if dose_log is None:
            dose_log = pd.DataFrame(episode_trajectory, columns=lbls)
        else:
            dose_log = pd.concat([dose_log, pd.DataFrame(episode_trajectory, columns=lbls)], axis=0)

        #the last reward tells recover or die
        fin_r = episode_trajectory[episode_trajectory.shape[0]-1, 0]        
        live_or_die = int(fin_r/abs(fin_r) if fin_r != 0 else 0)
        run_reps_stats.append(live_or_die)
        print('/',end='') # Keep from getting bored. 
    # record of outcomes for this dose
    run_stats.update({str(round(the_dose,2)): run_reps_stats})
    # Save the trajectories history for each episode, for each dose level. 
    dose_log.to_csv(f'{output_dir}/trajectory{unique_dt}__{the_dose}.csv')  
    print('\nDose\tSecs\n',the_dose, round(time.time() - start_time))
    bg_env.close()
    # return {"dose": [the list of outcomes]}
    return run_stats

#### MAIN ########################################################################
gym_major_version = re.findall(r'\.\d+\.', gym.__version__)[0].strip('.')
if int(gym_major_version) < 26:
    print(f'WARNING: gym version {gym_major_version} should be at least 26.')
    
if RUN_FLOW:
    mlflow.set_experiment("BogoEnv-iterAcc_"+unique_dt)
    mlflow_run = mlflow.start_run()
    mlflow.log_param('Reps', REPS)
start_time = time.time()
output_dir = 'output'+unique_dt
print(f'Creating folder {output_dir}')
os.makedirs(output_dir)

init_q()

#vrun_stats = dict()

# run_list = Parallel(n_jobs=8)(delayed(run_reps)(d/10.0) for d in range(0, 15))
# run_stats = {}
# for d in run_list:
#     run_stats.update(d)
# # One column for each dose level, rows are repetitions
# df_run_stats = pd.DataFrame(run_stats)
# df_run_stats = df_run_stats.applymap(lambda k: 1 if k==1 else 0).mean(axis=0)
# print('\nstats:\n',df_run_stats)   # Survival fraction 
# dfrs = df_run_stats.to_dict()
# tk =run_stats.keys()
# rs = {k: dfrs[k] for k in tk}
# # if RUN_FLOW: mlflow.log_metrics(rs)
# dose_response_fn = f'run_stats_{unique_dt}.csv'
# df_run_stats.to_csv(dose_response_fn)
if RUN_FLOW:
    mlflow.log_artifact(dose_response_fn)
    mlflow.end_run()
