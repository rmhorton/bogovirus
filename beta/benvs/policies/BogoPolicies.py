# BogoPolicies.py
#
# 13 March 2023

import os, re, sys
import math, random
from pathlib import Path
import pandas as pd
# print(Path.cwd())
# sys.path.append('./RL_offline/')
# from envs_beta.BogoBetaEnv import BogoBetaEnv
# from BogoBetaEnv import * 
VERBOSE = False

class BogoPolicies: # (BogoBetaEnv):
    'since the policy knows the cohort we can use cohort as  surrogate for grid search over e.g. dose. '
    
    def __init__(self, **params) -> None:    #TODO use **args instead of params
        super().__init__()
        # Settings that may vary at the patient or other levels,
        # not a function of state. 
        self.policy_params = params    # A dict Used for other possible customizations. E.G. max 
        self.alpha_new = params['alpha']  
        self.alpha_rate = params['rate']
        self.epsilon = params['epsilon']
        
    def alpha_iterator(self):
        'Call this each time to generate a descending series'

        self.alpha_new = self.alpha_rate * self.alpha_new
        return self.alpha_new
    
    def choose_epsilon_greedy(self, Q, old_state):
        ''
        #Choose the best "old" action 
        a_star = old_state.index[old_state.argmax()]
        # Randomize the choice of action TODO
        a_random_action = random.choice(old_state.index)
        if random.random() < self.epsilon:
            a_star = a_random_action
        return a_star
        
    def run_epsilon_greedy_policy(self, yesterday, today):
        'Use the Q matrix to select an action'
        # Note, the Q matrix is passed in with the yesterday states.
        current_Q = yesterday['Q']
        # maximize the Q value for the current states
        old_state = current_Q.loc[yesterday['severity'],:]
        new_state = current_Q.loc[today['severity'],:]
        max_Q = max(new_state)+ today['reward']
        # Compute the update to the Q matrix
        a_star = self.choose_epsilon_greedy(current_Q, old_state)
        update = self.alpha_new * \
            ( max_Q - old_state[a_star] )
        if VERBOSE:
            print(f'update: {update:.3g}, {self.alpha_new:.3g}')
        # Update the Q matrix
        current_Q.loc[yesterday['severity'], a_star] += update
        today['Q']= current_Q
        self.alpha_iterator()
        return a_star
        
    def const_policy(self, yesterday, today):  # default policy
        max_dose = self.policy_params['max_dose']    # Use this to scale dose 
        max_cohort= self.policy_params['max_cohort']
        dose = today['cohort'] * max_dose /  max_cohort
        return dose

    def standard_of_care_policy(self, yesterday, today):  # default policy
        # depends on: today's severity
        random_dose = self.my_rng.uniform( low=0.0, high=self.MAX_DOSE )
        severity_dependent_dose = random_dose * today['severity'] / self.SEVERITY_CEILING
        return math.floor(10 * severity_dependent_dose)/10 # rounded down to the nearest tenth

    def completely_random_policy(self, yesterday, today):
        # does not depend on anything
        dose = math.floor(10 * self.my_rng.uniform( low=0.0, high=self.MAX_DOSE ))/10
        return dose


    def dose_cohort_policy(self, yesterday, today):
        cohort = today['cohort']
        # print(f'c {cohort}')
        dose = self.MAX_DOSE * cohort / (self.num_cohorts - 1)
        return dose
    
if __name__ == '__main__':
    x = BogoPolicies(alpha=0.5, rate=0.9, max_dose=10, max_cohort=10)
    today = {'cohort':0}
    print('dose:', x.const_policy(None, today))
