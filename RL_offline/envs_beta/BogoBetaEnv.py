# gym environment for the bogovirus using the new, true beta simulator 
#  JMA 6 March 2023

import math, os, re, sys
import numpy as np
import pandas as pd
from numpy.random import default_rng
import datetime as dt
# import gym
# from gym.spaces import Box, 
# gym-like objects
Dict = dict

class Box(object):
    'To test if a value fits in an interval'
    def __init__(self, low = 0, high= 1, shape=(1,), dtype=np.float32) -> None:
        'shape is not used.'
        self.low = low
        self.high = high
        
    def not_within(self, v):
        'test if the value v is outside the interval'
        if v > self.high:
            return 'high'
        elif v < self.low:
            return 'low'
        else:
            return False
        
        

sys.path.append('./beta/')
# Get the set of policies

# import beta_simulation as sm

SEED = None       # Use a random seed each run
CONST_DOSE  = 0.7 # For test
VERBOSE = True

def sigmoid(x):
    # starts at 1, goes to zero
    return 1 - 1/(1 + math.exp(-(x - 0.6)/0.1))

class BogoBetaEnv(object):
    'An environment class derived from the gym environment, for one patient episode.'
    
    my_rng = default_rng(seed=None)

    NUM_COHORTS = 1 
    MAX_INFECTION = 150
    MAX_DOSE = 3.0  # we want doses from 0 to 1.5 in 0.1 increments
    SEVERITY_CEILING = 125; # Max expected severity.
    MAX_DAYS = 100 
    SAMPLES = 10

    def __init__(self) -> None:
        'Call this once, and reuse it for all episodes'
        # self.n_neighbors = N_NEIGHBORS 
        # self.render_mode = None
        # fix the type warnings for these

        self.action_space = Dict(
            {"Dose": Box(low=0.0, high=self.MAX_DOSE, shape=(1,), dtype=np.float32)}
        )
        self.observation_space = Dict(
            {"infection": Box(low=0, high=self.MAX_INFECTION, shape=(1,), dtype=np.float32), 
             "severity": Box(low=0, high=self.SEVERITY_CEILING, shape=(1,), dtype=np.float32),
             "cum_drug": Box(low=0, high=self.MAX_DOSE, shape=(1,), dtype=np.float32)}
        )
        # Immediate rewards
        self.one_day = -1
        self.recover = 100
        self.die = -200
        # Since the simulator doesn't observe the full state we keep it internal to the objec
        self.today = ModuleNotFoundError
        
    def test_v(self, the_var, v):
        'Is the var in range?'
        space = self.observation_space.get(the_var, None)
        if space is not None:
            is_not = space.not_within(v)
            if is_not:
                print(f'Error {the_var}:{round(v,3)} is {is_not}')
        return v
        
    ### Model local models
        
    def new_patient(self, patient_id,):
        'Return a dict of a patient with an initial infection and its severity.'
        self.stage = 0               # Not a state variable, but the sim tracks it
        self.patient_results = []
        today =  {
                'patient_id': patient_id,             # None of these variable are part of the state
                'cohort': patient_id % self.NUM_COHORTS,   # 
                'day_number': self.stage,             # The stage
                # e.g. An array of length 1, of random integers, 20 <= rv < 40
                'infection': self.my_rng.integers(low=20, high=40, size=1)[0],
                'severity': self.my_rng.integers(low=10, high=30, size=1)[0],
                'cum_drug' : 0,
                'outcome':None,
                'efficacy': 0,
                'drug': 0,
                'reward' : 0
            }
        self.today = today
        return today
        
    def get_infection(self, yesterday):
        # depends on: infection_prev
        progression = self.my_rng.integers(low=0, high=10, size=1)[0]
        return self.test_v('infection', yesterday['infection'] + progression)
    
    def get_severity(self, yesterday):
        # depends on: yesterday's severity, infection, & efficacy
        noise = self.my_rng.normal(loc=0, scale=1, size=1)[0]
        severity_next = yesterday['severity'] * 1.1 + \
                        yesterday['infection'] * 0.1 - \
                        yesterday['efficacy'] + \
                        noise
        return self.test_v('severity', severity_next)
    
    def get_cum_drug(self, yesterday, today, proportion=0.7):
        'Mix the current drug level and the current dose in fixed proportion.'
        # depends on: cum_drug_prev, drug
        noise = self.my_rng.normal(loc=0, scale=0.01, size=1)[0]  # !!! surprisingly sensitive !!!
        r = proportion + noise  # # larger value responds more slowly to changes in dose
        return self.test_v('cum_drug',yesterday['cum_drug'] * r + today['drug'] * (1 - r))
    
    def get_outcome(self, today):
        # depends on today's severity, infection, and cum_drug
        # possible outcomes: die, recover, none
        noise = self.my_rng.normal(loc=0, scale=0.1, size=1)[0]
        mortality_threshold = 1.0 + noise # my_rng.uniform(low=0.9, high=1.1)
        if (today['severity']/self.SEVERITY_CEILING > mortality_threshold):
            return 'die'
        elif today['infection'] >= 100:
            return 'recover'
        else:
            return None
    
    def get_efficacy(self, today):
        # depends on today's drug and cum_drug
        # The amount by which severity will be reduced. 
        # Maybe this shold be a proportion not a fixed amount? Severity can be negative this way.
        noise = self.my_rng.normal(loc=0, scale=1, size=1)[0]
        # efficacy = sigmoid( 12 * today['drug'] * today['cum_drug'] + noise )
        efficacy = 12 * today['drug'] * sigmoid( today['cum_drug'] ) + noise
        return efficacy
    
    def reward(self, today):
        'reward shaping for the outcome.'
        if today['outcome'] == 'die':
            return self.die
        elif today['outcome'] == 'recover':
            return self.recover
        else:
            return self.one_day

### Simulation  Environment functions

    def reset(self, id_serial, sd=SEED, options=None) -> dict:
        'Initialize a patient -  episode'
        # Set state variables.
        # THe state is observable, so we use observation as the state. 
        # Of course for a constant policy observability is moot.
        self.stage = 0 
        self.today = self.yesterday = self.new_patient(patient_id=id_serial)
        #  Features for the predictor -- representing the current state. Only those features samples will be searched on. 

        info = {'stage': self.stage}   # Just a place to return additional info
                                       # that is not part of the state, e.g. for diagnostics
        return self.get_observation(), info
    
    def cycle(self, yesterday, day_number, policy):
        today = {
            'patient_id': yesterday['patient_id'],
            'cohort': yesterday['cohort'],
            'day_number': day_number,
        } 
        # Note, the order these are called matters.
        today['infection'] = self.get_infection(yesterday)
        today['severity']  = self.get_severity(yesterday)
        today['drug']      = policy(yesterday, today)
        today['cum_drug']  = self.get_cum_drug(yesterday, today)
        today['efficacy']  = self.get_efficacy(today)
        today['outcome']   = self.get_outcome(today)
        today['reward']    = self.reward(today)
        return today


    def step(self, policy):
        'Increment the state at each stage in an episode, and terminate on death or recovery.'
        # Call cycle
        self.stage += 1
        today = self.cycle(self.yesterday, self.stage, policy)
        self.today = self.yesterday = today
        info = {"stage": self.stage}
        # Return only those things the RL solver can see. 
        self.patient_results.append(today)
        return self.get_observation(), today['reward'],  (today['outcome'] is not None) , info
    
    def close(self):
        'Anything to finish up an episode'
        # Note - to get the temporal df needed for causal learning join each
        #        row with its previous row.  
        episode = pd.DataFrame(self.patient_results)
        cum_reward = episode.reward.sum()
        return episode, cum_reward

    def get_observation(self):
        'A convenience function to format the observable output '
        return {"Severity":self.today['severity']}
    
### Use this policy for test
def test_const_policy(yesterday, today):  # default policy
    dose = CONST_DOSE
    return dose
   

### a test run
def test_patient_run(env):
    ''
    # Create a patient with a random Id. 
    the_patient = BogoBetaEnv.my_rng.integers(low=0, high=100000, size=1)[0]
    observation, info = env.reset(id_serial= the_patient)
    print('\t', observation)
    for _ in range(BogoBetaEnv.MAX_DAYS):
        observation, reward, terminated, info = env.step(test_const_policy)
        if VERBOSE: 
            print(env.today)
        else:
            print(f'i: {info} # obs: {env.get_observation()}, R: {reward}, end? {terminated}' )
        if terminated:
            break

if __name__ == '__main__':
    
    # Run one episode 
    bogo_env = BogoBetaEnv()
    test_patient_run(bogo_env)
    episode_df, total_reward = bogo_env.close()
    print(f'DONE! - reward {total_reward}')
    