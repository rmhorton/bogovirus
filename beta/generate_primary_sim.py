# harness to run BogoEnv beta online environment
# from "runenv.py"
# JMA 7 March 2023
import os, re, sys, time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


sys.path.append('beta/benvs/online')
from BogoBetaEnv import BogoBetaEnv
# sys.path.append('beta/benvs/policies')
# from BogoPolicies import BogoPolicies


	
def one_patient_run(env, serial):
    'Run one patient episode.'
    # A column for each variable: stage, Drug-action, Infection, CumulativeDrug, Severity, reward 
    run_trajectory = []
    # Create a patient episode, and ignore this observation
    observation, info = env.reset(id_serial= serial)
    
    for step in range(BogoBetaEnv.MAX_DAYS):
        observation, reward, terminated, info = env.step(env.the_policy)
        if args.verbose: 
            print(env.today)
        else:
            # print(f'i: {info} # obs: {env.get_observation()}, R: {reward}, end? {terminated}' )
            run_trajectory.append(env.today)
        if terminated:
            break
    env.close()
    return pd.DataFrame(run_trajectory)

def file_w_ts(fn: str) -> str:
    ts = datetime.now().strftime('%j-%H-%M')
    print(f'Writing to file: {fn}_{ts}' )
    return f'{fn}_{ts}'

def run_with_policy(the_env: BogoBetaEnv):
    
    all_trajectories = [] # The results - the last record - in each trajectory 
    record_cnt = 0        # Data set size
    idx = 0               # patient Id
    for a_cohort in range(args.cohorts):
        # Adjust the policy 
        the_policy = BogoPolicies( max_dose=the_env.MAX_DOSE, max_cohort=args.cohorts).const_policy 
        the_env.the_policy = the_policy
        for a_sample in range(args.samples):
            # 
            one_trajectory = one_patient_run(the_env, idx)
            idx +=1
            run_outcome = one_trajectory.iloc[-1,:].to_dict()
            record_cnt += run_outcome['day_number']
            all_trajectories.append(run_outcome)
            if args.verbose:
                print(run_outcome)
            with open(file_w_ts(args.simulation_file)+'.csv', 'ab') as out_fd:
                if a_cohort == 0 and a_sample == 0:
                    one_trajectory.to_csv(out_fd, na_rep='survive', header=True, index=False)
                else:
                    one_trajectory.to_csv(out_fd, na_rep='survive', header=False, index=False)
                    
    all_trajectories = pd.DataFrame(all_trajectories)       
    with open(file_w_ts('summary_'+args.simulation_file)+'.csv', 'wb') as out_fd:
            all_trajectories.to_csv(out_fd, index=False)
            
    num_recovered = np.sum(all_trajectories.outcome == 'recover')
    num_died = np.sum(all_trajectories.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died. Recovery fraction: {num_recovered / (num_recovered + num_died):.3} ")
    print(f'{record_cnt} records.')
    
### MAIN ################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = 'What the program does')

    parser.add_argument('simulation_file',
                        default='output.parquet',
                        help='output file')   # positional argument
    parser.add_argument('-d', '--discretize',
                        help = 'Round all observable values to multiples of 10',
                        action='store_true')            # option that takes a numeric value
    parser.add_argument('-c', '--cohorts',
                        help = 'number of cohorts (1)',
                        default = 1,
                        type=int)           
    parser.add_argument('-s', '--samples',
                        help = 'samples, i.e. patients per cohort (10)',
                        default = 10,
                        type=int)            
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    # values can be found in a named tuple: args.filename, args.count, args.verbose
    params = dict(discretize=False,
                SAMPLE_SAVE=False, 
                        samples=2)
    params.update(vars(args)) 
    st = time.time()
    bogo_env = BogoBetaEnv(**params) # None, NUM_COHORTS= args.cohorts, discretize=args.discretize)    # we set the policy later.
    run_with_policy(bogo_env)
    print(f'Done in {time.time() - st:.2} seconds!')
