# harness to run BogoEnv beta online environment
# from "generate_primary_sim.py"
# JMA 17 June 2023
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

    
###  Functions ###################
def file_w_ts(dir_n: str, z:str, params:dict, suffix: str) -> Path:
    vars = '_'.join([f'{x}-{y}' for x, y in params.items() if type(y) is int])
    ts = f'{z}{vars}_' + datetime.now().strftime('%j-%H-%M') + suffix
    return Path(dir_n) / Path(ts)
    
def one_patient_run(env, serial):
    'Run one patient episode.'
    # A column for each variable: stage, Drug-action, Infection, CumulativeDrug, Severity, reward 
    run_trajectory = []
    # Create a patient episode, and ignore this observation
    observation, info = env.reset(id_serial= serial)
    
    for step in range(env.params['max_days']):
        observation, reward, terminated, info = env.step(env.the_policy)
        if env.params['verbose']: 
            print(env.today)
        else:
            pass
            # print(f'i: {info} # obs: {env.get_observation()}, R: {reward}, end? {terminated}' )
        run_trajectory.append(env.today)
        if terminated:
            break
    env.close()
    return pd.DataFrame(run_trajectory)

def run_policy_over_grid(the_env: BogoBetaEnv, const_space, change_space):
    
    all_trajectories = [] # The results - the last record - in each trajectory 
    record_cnt = 0        # Data set size
    idx = 0               # patient Id
    for const_value in const_space:
        for change_value in change_space:
            a_cohort = (const_value, change_value) 
            print(f'\tcohort: {a_cohort}')
            # Adjust the policy 
            #the_policy = BogoPolicies(**params).linear_change_policy
            the_env.policies.const_dose = const_value
            the_env.policies.daily_change = change_value
            the_env.the_policy = the_env.policies.linear_change_policy 
            for a_sample in range(the_env.params['samples']):
                # 
                one_trajectory = one_patient_run(the_env, idx)
                idx +=1
                run_outcome = one_trajectory.iloc[-1,:].to_dict()
                run_outcome['cohort'] = a_cohort
                record_cnt += run_outcome['day_number']
                all_trajectories.append(run_outcome)
                if the_env.params['verbose']:
                    print(run_outcome)
                if the_env.params['SAMPLE_SAVE']:
                    with open(file_w_ts(the_env.params['simulation_dir']),'T', params, '.csv', 'ab') as out_fd:
                        if idx == 1:
                            one_trajectory.to_csv(out_fd, na_rep='survive', header=True, index=False)
                        else:
                            one_trajectory.to_csv(out_fd, na_rep='survive', header=False, index=False)
                    
    all_trajectories = pd.DataFrame(all_trajectories)
    if the_env.params.get('simulation_dir', None) :       
        with open(file_w_ts(the_env.params['simulation_dir'], 'T', the_env.params, '.csv'), 'wb') as out_fd:
            all_trajectories.to_csv(out_fd, index=False)
    #normalizedQ = QN.Q/QN.N
        Qfilename = file_w_ts(the_env.params['simulation_dir'], 'Q', the_env.params, '.csv')
        # with open(Qfilename, 'wb') as out_fd:
        #     QN.Q.to_csv(out_fd)
        # print(f'\nWrote {Qfilename}')            
    num_recovered = np.sum(all_trajectories.outcome == 'recover')
    num_died = np.sum(all_trajectories.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died. Recovery fraction: {num_recovered / (num_recovered + num_died):.3} ")
    print(f'{record_cnt} records.')
    
### MAIN ################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = 'What the program does')

    parser.add_argument('simulation_dir',
                        nargs='?',            # make it optional
                        default='stage',
                        action = 'store',
                        help='output directory')        # positional argument
    parser.add_argument('-d', '--discretize',
                        help = 'Round all observable values to multiples of 10',
                        action='store_true')            # option that takes a numeric value
    # parser.add_argument('-c', '--cohorts',
    #                     help = 'number of cohorts (1)',
    #                     default = 1,
    #                     type=int)           
    # parser.add_argument('-o', '--dose',
    #                     help = 'average daily dose (0.7)',
    #                     default = 0.7,
    #                     type=float)        # 
    parser.add_argument('-m', '--mid_day',
                        help = 'day when the average dose is met. (7)',
                        default = 7.0,
                        type=float)        # 
    # parser.add_argument('-i', '--daily_change',
    #                     help = 'incremental increase in daily dose (0)',
    #                     default = 0.0,
    #                     type=float)    
    parser.add_argument('-s', '--samples',
                        help = 'samples, i.e. patients per cohort (1)',
                        default = 1,
                        type=int)            
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    # values can be found in a named tuple: paramsfilename, paramscount, args.verbose
    # convert the named tuple to a dict
    
        # The parameters are passed to the Policy object 
    params = dict(discretize=False,
                SAMPLE_SAVE=False, 
                        samples=2, 
                        mid_day=6.7) 
                        # daily_change= 0.01, 
                        # const_dose=0.4)
    
    params.update(vars(args)) 
    # params['SAMPLE_SAVE'] = False          # Spend the IO cycles to save every step in episodes

    st = time.time()
    bogo_env = BogoBetaEnv(**params)
    const_dose = np.linspace(0.6, 0.8, 21)
    daily_dose_change = np.linspace(-0.2, 0.15, 21)
    run_policy_over_grid(bogo_env, const_dose, daily_dose_change)
    print(f'Done in {time.time() - st:.2} seconds!')
