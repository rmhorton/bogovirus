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

class BogoPolicies: # (BogoBetaEnv):
    
    def __init__(self, params = 0) -> None:
        super().__init__()
        # Settings that may vary at the patient or other levels,
        # not a function of state. 
        self.policy_params = params
        
    def const_policy(self, yesterday, today):  # default policy
        dose = self.policy_params
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
