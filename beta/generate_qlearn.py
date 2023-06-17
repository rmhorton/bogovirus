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

    parser.add_argument('simulation_dir',
                        nargs='?',            # make it optional
                        default='stage',
                        action = 'store',
                        help='output directory')        # positional argument
    parser.add_argument('-d', '--discretize',           # TODO parameterize the discretization
                        action='store_true')            # option that takes a numeric value
    parser.add_argument('-c', '--cohorts',              # Also used to pace periodic learning steps
                        help = 'number of cohorts (1)',
                        default = 100,
                        type=int)           
    parser.add_argument('-s', '--samples',              # If Q converges before this limit is met, all samples will not be run
                        help = 'samples, i.e. patients per cohort (10)',
                        default = 10,
                        type=int)  
    parser.add_argument('-a', '--alpha',
                        help = 'learning rate (0.10)',
                        default = 0.10,
                        type=float)            
    parser.add_argument('-r', '--decay',
                        help = 'alpha decay per cohort iteration (0.99)',
                        default = 0.99,
                        type=float)            
    parser.add_argument('-e', '--epsilon',
                        help = 'greedy exploration probability (0.2)',
                        default = 0.2,
                        type=float)                      
    parser.add_argument('-v', '--verbose',
                        default = False,
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    # values can be found in a named tuple: args.filename, args.count, args.verbose
   
# ALPHA = 0.4
# RATE = 0.999
# EPSILON = 0.1
SAMPLE_SAVE = False          # Spend the IO cycles to save every step in episodes

def file_w_ts(dir_n: str, z:str, a:int, d:int, e:int, suffix: str) -> Path:
    ts = f'{z}{a}_{d}_{e}_' + datetime.now().strftime('%j-%H-%M') + suffix
    return Path(dir_n) / Path(ts)

class QLearner (object):
    'The Q matrix and related functions'
    
    def __init__(self, env, a) -> None:
        self.Q, self.N = self.init_q_matrix(env)
        # Estimate the survival rate from the sequence of episodes. 
        self.running_survival = 0.0
        self.decay = 0.99 # a.decay
        

    def init_q_matrix(self, env):
        'Create a Q matrix and its shadow count matrix'
        ACTIONS = [round(z,1) for z in np.linspace(0, env.MAX_DOSE, 13)]
        OBSERVATIONS = [0]   # [round(z,0) for z in np.linspace(0, env.SEVERITY_CEILING, 13)]
        Q_DEFAULT = 0
        Q = Q_DEFAULT * np.ones((len(OBSERVATIONS), len(ACTIONS)))
        Q = pd.DataFrame(Q, index=OBSERVATIONS, columns=ACTIONS)
        N =  np.zeros((len(OBSERVATIONS), len(ACTIONS)))
        N = pd.DataFrame(N, index=OBSERVATIONS, columns=ACTIONS)
        return Q, N
    
### Iterations ##############################################

def one_patient_run(QN, env, serial):
    'Run one patient episode.'
    # A column for each variable: stage, Drug-action, Infection, CumulativeDrug, Severity, reward 
    run_trajectory = []
    # Create a patient episode, and ignore this observation
    observation, info = env.reset(id_serial= serial)
    
    for step in range(BogoBetaEnv.MAX_DAYS):
        observation, reward, terminated, info = env.step(env.the_policy, QN)
        QN = info['QN']    # return Q from env.step()
        if args.verbose: 
            print(env.patient_results[-1])
        else:
            pass
            # print(f'i: {info} # obs: {env.get_observation()}, R: {reward}, end? {terminated}' )
        #   run_trajectory.extend(env.patient_results)
        if terminated:
            break
    # Update the estimated survival rate
    # TODO move the alpha decay to this step 
    episode, _ = env.close(QN)
    return QN, episode
    
def run_with_policy(the_env: BogoBetaEnv):
    
    all_trajectories = [] # The results - the last record - in each trajectory 
    record_cnt = 0        # Data set size
    idx = 0               # patient Id
    # The Q matrix is shared among all patient episodes, so it is initialized here.
    QN = QLearner(the_env, args)   # TODO merge these two parameter objects 
    for a_cohort in range(args.cohorts):
        # Adjust the policy 

        the_policy = the_env.policies.run_const_greedy_policy  # run_epsilon_greedy_policy
        # the_env.the_policy = the_policy
        for a_sample in range(args.samples):
            Q_checkpoint = QN.Q.sum().sum()
            # Pass Q explicity  
            QN, one_trajectory= one_patient_run(QN, the_env, idx)
            # Update the learning rate.
            # the_env.the_policy.alpha_iterator() TODO
            idx +=1
            run_outcome = one_trajectory.iloc[-1,:].to_dict()
            record_cnt += run_outcome['day_number']
            all_trajectories.append(run_outcome)
            if args.verbose:
                print(run_outcome)
            if SAMPLE_SAVE:
                with open(file_w_ts(args.simulation_dir, 'S', args.alpha, args.decay, args.epsilon, '.parquet'), 'ab') as out_fd:
                    if a_cohort == 0 and a_sample == 0:
                        one_trajectory.to_parquet(out_fd, index=False)
                    else:
                        one_trajectory.to_csv(out_fd, na_rep='survive', header=False, index=False)
            sample_Q_inc = Q_checkpoint - QN.Q.sum().sum()
            print(f'Q delta: {sample_Q_inc:.3g} Q sum: {QN.Q.sum().sum():.3}')
            if abs(sample_Q_inc) < 1e-6:
                break 
        # Use the for - else magic to break outer loop!
        else:
            continue # continue inner loop
        break        # break the outer loop
        
    all_trajectories = pd.DataFrame(all_trajectories)       
    with open(file_w_ts(args.simulation_dir, 'A', args.alpha, args.decay, args.epsilon, '.csv'), 'wb') as out_fd:
        all_trajectories.to_csv(out_fd, index=False)
    #normalizedQ = QN.Q/QN.N
    Qfilename = file_w_ts(args.simulation_dir, 'Q', args.alpha, args.decay, args.epsilon, '.csv')
    with open(Qfilename, 'wb') as out_fd:
        QN.Q.to_csv(out_fd)
    print(f'\nWrote {Qfilename}')
            
    num_recovered = np.sum(all_trajectories.outcome == 'recover')
    num_died = np.sum(all_trajectories.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died. Recovery fraction: {num_recovered / (num_recovered + num_died):.3} ")
    print(f'{record_cnt} records.')
    return num_recovered, num_died
    
### MAIN ################################################################################
st = time.time()

policies = BogoPolicies(# max_dose=the_env.MAX_DOSE,    #TODO pass this as an arg.  
                            max_cohort=args.cohorts, 
                            alpha=args.alpha, 
                            rate=args.decay,
                            epsilon=args.epsilon) 
bogo_env = BogoBetaEnv(policies, NUM_COHORTS= args.cohorts, discretize=args.discretize)    # we set the policy later.
run_with_policy(bogo_env)
print(f'alpha_final: {policies.alpha_new}, epsilon final: {policies.epsilon}')
print(f'Done in {time.time() - st:.2} seconds!')
