# harness to run Q learning on the beta online environment
# JMA 8 June 2023
import os, re, sys, time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


sys.path.append('beta/benvs/online')
from BogoBetaEnv import BogoBetaEnv
sys.path.append('beta/benvs/policies')
from BogoPolicies import BogoPolicies

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = 'What the program does')

    parser.add_argument('simulation_file',
                        nargs='?',            # make it optional
                        default='output.parquet',
                        action = 'store',
                        help='output file')   # positional argument
    parser.add_argument('-d', '--discretize',
                        help = 'Round all observable values to multiples of 10',
                        default = True,
                        action='store_true')            # option that takes a numeric value
    parser.add_argument('-c', '--cohorts',
                        help = 'number of cohorts (1)',
                        default = 100,
                        type=int)           
    parser.add_argument('-s', '--samples',
                        help = 'samples, i.e. patients per cohort (10)',
                        default = 10,
                        type=int)            
    parser.add_argument('-v', '--verbose',
                        default = False,
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    # values can be found in a named tuple: args.filename, args.count, args.verbose
   
ALPHA = 0.4
RATE = 0.999
EPSILON = 0.1

def file_w_ts(fn: str) -> str:
    ts = datetime.now().strftime('%j-%H-%M')
    return f'{fn}_{ts}'

def init_q_matrix(env):
    'Create a Q matrix'
    ACTIONS = [round(z,1) for z in np.linspace(0, env.MAX_DOSE, 13)]
    OBSERVATIONS = [round(z,0) for z in np.linspace(0, env.SEVERITY_CEILING, 13)]
    Q_DEFAULT = 0
    Q = Q_DEFAULT * np.ones((len(OBSERVATIONS), len(ACTIONS)))
    Q = pd.DataFrame(Q, index=OBSERVATIONS, columns=ACTIONS)
    # Q is a 10 x 12 matrix of Q values. 
    # Q is initialized to 0. 
    # Q is bounded by -100 and 100. TODO 
    # Q is indexed by the observation and action value
    return Q

def one_patient_run(Q, env, serial):
    'Run one patient episode.'
    # A column for each variable: stage, Drug-action, Infection, CumulativeDrug, Severity, reward 
    run_trajectory = []
    # Create a patient episode, and ignore this observation
    observation, info = env.reset(id_serial= serial)
    
    for step in range(BogoBetaEnv.MAX_DAYS):
        observation, reward, terminated, info = env.step(Q, env.the_policy)
        Q = info['Q']
        if args.verbose: 
            print(env.patient_results[-1])
        else:
            pass
            # print(f'i: {info} # obs: {env.get_observation()}, R: {reward}, end? {terminated}' )
        #   run_trajectory.extend(env.patient_results)
        if terminated:
            break
    episode, _ = env.close()
    return Q, episode
    
def run_with_policy(the_env: BogoBetaEnv):
    
    all_trajectories = [] # The results - the last record - in each trajectory 
    record_cnt = 0        # Data set size
    idx = 0               # patient Id
    # The Q matrix is shared among all patient episodes, so it is initialized here.
    Q = init_q_matrix(the_env)
    for a_cohort in range(args.cohorts):
        # Adjust the policy 
        the_policy = BogoPolicies(max_dose=the_env.MAX_DOSE, 
                                  max_cohort=args.cohorts, 
                                  alpha=ALPHA, 
                                  rate=RATE,
                                  epsilon=EPSILON).run_epsilon_greedy_policy 
        the_env.the_policy = the_policy
        for a_sample in range(args.samples):
            Q_checkpoint = Q.sum().sum()
            # Pass Q explicity  
            Q, one_trajectory= one_patient_run(Q, the_env, idx)
            # Update the learning rate.
            # the_env.the_policy.alpha_iterator() TODO
            idx +=1
            run_outcome = one_trajectory.iloc[-1,:].to_dict()
            record_cnt += run_outcome['day_number']
            all_trajectories.append(run_outcome)
            if args.verbose:
                print(run_outcome)
            with open(file_w_ts(args.simulation_file)+'.parquet', 'ab') as out_fd:
                if a_cohort == 0 and a_sample == 0:
                    one_trajectory.to_parquet(out_fd, index=False)
                else:
                    one_trajectory.to_csv(out_fd, na_rep='survive', header=False, index=False)
            print(f'Q delta: {Q_checkpoint - Q.sum().sum():.3} Q sum: {Q.sum().sum():.3}') # Q min: {Q.min().min():.3} Q mean: {Q.mean().mean():.3}')

    all_trajectories = pd.DataFrame(all_trajectories)       
    with open(file_w_ts('summary_'+args.simulation_file)+'.csv', 'wb') as out_fd:
            all_trajectories.to_csv(out_fd, index=False)
            
    with open(file_w_ts('Q_'+args.simulation_file)+'.csv', 'wb') as out_fd:
            Q.to_csv(out_fd)
            
    num_recovered = np.sum(all_trajectories.outcome == 'recover')
    num_died = np.sum(all_trajectories.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died. Recovery fraction: {num_recovered / (num_recovered + num_died):.3} ")
    print(f'{record_cnt} records.')
    
### MAIN ################################################################################
st = time.time()

bogo_env = BogoBetaEnv(None, NUM_COHORTS= args.cohorts, discretize=args.discretize)    # we set the policy later.
run_with_policy(bogo_env)
print(f'Done in {time.time() - st:.2} seconds!')
