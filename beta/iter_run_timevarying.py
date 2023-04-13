# harness to run BogoBetaEnv
# with the time-varying policy
# JMA 22 March 2023
import time
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parent))
import envs_beta
# import mlflow

EPISODE_LEN = 100
REPS = 100     # Number of episodes to average 
unique_dt = time.strftime( 'T%j-%H-%M-%S')


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
            # mlflow.log_metrics(observation)
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
    
# mlflow.set_experiment("BogoEnv-iterAcc_"+unique_dt)

# mlflow_run = mlflow.start_run()

# mlflow.log_param('Reps', REPS)
start_time = time.time()
output_dir = 'output'+unique_dt
print(f'Creating folder {output_dir}')
os.makedirs(output_dir)

#vrun_stats = dict()

run_list = Parallel(n_jobs=8)(delayed(run_reps)(d/10.0) for d in range(0, 15))
run_stats = {}
for d in run_list:
    run_stats.update(d)
# One column for each dose level, rows are repetitions
df_run_stats = pd.DataFrame(run_stats)
df_run_stats = df_run_stats.applymap(lambda k: 1 if k==1 else 0).mean(axis=0)
print('\nstats:\n',df_run_stats)   # Survival fraction 
dfrs = df_run_stats.to_dict()
tk =run_stats.keys()
rs = {k: dfrs[k] for k in tk}
# mlflow.log_metrics(rs)
dose_response_fn = f'run_stats_{unique_dt}.csv'
df_run_stats.to_csv(dose_response_fn)
# mlflow.log_artifact(dose_response_fn)
# mlflow.end_run()
