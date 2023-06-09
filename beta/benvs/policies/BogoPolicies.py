# BogoPolicies.py
#
# 13 March 2023

import os, re, sys
import math
from pathlib import Path
# print(Path.cwd())
# sys.path.append('./RL_offline/')
# from envs_beta.BogoBetaEnv import BogoBetaEnv
# from BogoBetaEnv import * 
VERBOSE = True

class BogoPolicies: # (BogoBetaEnv):
    'since the policy knows the cohort we can use cohort as  surrogate for grid search over e.g. dose. '
    
    def __init__(self, **params) -> None:    #TODO use **args instead of params
        super().__init__()
        # Settings that may vary at the patient or other levels,
        # not a function of state. 
        self.policy_params = params    # A dict Used for other possible customizations. E.G. max 
        
    def alpha_iterator(self):
        'Call this each time to generate a descending series'
        alpha_new = self.policy_params['alpha']  
        alpha_rate = self.policy_params['rate']
        alpha_new = alpha_rate * alpha_new
        return alpha_new
    
        
    def run_epsilon_greedy_policy(self, yesterday, today):
        'Use the Q matrix to select an action'
        # Note, the current Q matrix is passed with the yesterday and today states.
        current_Q = self.today['Q']
        # maximize the Q value for the current states
        # This is V(s)
        max_Q = current_Q.max(axis=1) + today['reward']
        # Compute the update to the Q matrix
        update = self.policy_params['alpha'] * \
            ( max_Q - yesterday['Q'])
        if VERBOSE:
            print(f'update: {update}')
        # Update the Q matrix
        yesterday['Q'] += update
        # Update the learning rate.
        self.policy_params['alpha'] = self.alpha_iterator()
        
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
    x = BogoPolicies()
    print('dose:', x.const_policy(None, None))
