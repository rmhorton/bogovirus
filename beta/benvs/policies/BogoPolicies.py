# BogoPolicies.py
#
# 13 March 2023

import os, re, sys
import math, random
from pathlib import Path
import pandas as pd
from numpy.random import default_rng

params = {'verbose': True}

CUM_DRUG = True


class BogoPolicies: # (BogoBetaEnv):
    'since the policy knows the cohort we can use cohort as  surrogate for grid search over e.g. dose. '
    
        
    def __init__(self, **params) -> None:    #TODO use **args instead of params
        super().__init__()
        # Settings that may vary at the patient or other levels,
        # not a function of state. 
        self.my_rng = default_rng(0)
        self.policy_params = params    # A dict Used for other possible customizations. E.G. max 
        self.max_dose = params.get('max_dose', None)    # Use this to scale dose 
        self.max_cohort= params.get('max_cohort', None)
        # Q learning params
        self.alpha_new = params.get('alpha', None) 
        self.alpha_rate = params.get('rate', None) 
        self.epsilon = params.get('epsilon', None) 
        # Customizations for a linear change policy
        self.mid_day = params.get('mid_day', None) 
        self.daily_change = params.get('daily_change', None)
        self.const_dose = params.get('const_dose', None)
        
    def alpha_iterator(self):
        'Call this each time to generate a descending series'

        self.alpha_new = self.alpha_rate * self.alpha_new
        self.epsilon = self.alpha_rate * self.epsilon
        return self.alpha_new, self.epsilon
    
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
        current_Q = yesterday['QN'].Q
        current_N = yesterday['QN'].N
        # maximize the Q value for the current states
        old_state = current_Q.loc[yesterday['severity'],:]
        new_state = current_Q.loc[today['severity'],:]
        max_Q = max(new_state)+ today['reward']
        # Pick a policy
        a_star = self.choose_epsilon_greedy(current_Q, old_state)
        # Q learning update: compute the update to the Q matrix
        update = self.alpha_new * \
            ( max_Q - old_state[a_star] )
        # update N, the normalizing factor for Q. 
        current_N.loc[yesterday['severity'], a_star] += self.alpha_new
        if params['verbose']:
            print(f'update: {update:.3g}, {self.alpha_new:.3g}')
        # Update the Q matrix
        current_Q.loc[yesterday['severity'], a_star] += update
        today['QN'].Q= current_Q
        today['QN'].N= current_N
        self.alpha_iterator()
        return a_star
    
    def run_epsilon_greedy_on_cum_drug(self, yesterday, today, proportion=0.7):
        'Use the Q matrix where cum drug is the observation.'
        # Note, the Q matrix is passed in with the yesterday states.
        current_Q = yesterday['QN'].Q
        current_N = yesterday['QN'].N
        # maximize the Q value for the current states
        old_state = current_Q.loc[yesterday['cum_drug'],:]
        # Pick a policy
        a_star = self.choose_epsilon_greedy(current_Q, old_state)
        # TODO do cum_drug state  up date here. 
        noise = self.my_rng.normal(loc=0, scale=0.1, size=1)[0]  # !!! surprisingly sensitive !!!
        r = proportion + noise
        new_state = old_state * r + a_star * ( 1-r)
        #new_state = self.test_v('cum_drug', yesterday['cum_drug'] * r + a_star * ( 1-r))
        # new_state = current_Q.loc[today['cum_drug'],:]
        max_Q = max(new_state)+ today['reward']
        # Q learning update: compute the update to the Q matrix
        update = self.alpha_new * \
            ( max_Q - old_state[a_star] )
        # update N, the normalizing factor for Q. 
        current_N.loc[yesterday['cum_drug'], a_star] += self.alpha_new
        if params['verbose']:
            print(f'update: {update:.3g}, {self.alpha_new:.3g}')
        # Update the Q matrix

        current_Q.loc[yesterday['cum_drug'], a_star] += update
        today['QN'].Q= current_Q
        today['QN'].N= current_N
        self.alpha_iterator()
        return a_star
    
    def run_const_greedy_policy(self, yesterday, today):
        'A test policy that has no observation, so it is const over all states.'
        # Use the Q matrix to select an action'
        # Note, the Q matrix is passed in with the yesterday states.
        current_Q = yesterday['QN'].Q
        current_N = yesterday['QN'].N
        # maximize the Q value for the current state
        old_state = current_Q.loc[0,:]
        new_state = current_Q.loc[0,:]
        max_Q = max(new_state)+ today['reward']
        # Pick a policy
        a_star = self.choose_epsilon_greedy(current_Q, old_state)
        # Q learning update: compute the update to the Q matrix
        update = self.alpha_new * \
            ( max_Q - old_state[a_star] )
        # update N, the normalizing factor for Q. 
        current_N.loc[0, a_star] += self.alpha_new
        if params['verbose']:
            print(f'update: {update:.3g}, {self.alpha_new:.3g}')
        # Update the Q matrix
        current_Q.loc[0, a_star] += update
        today['QN'].Q= current_Q
        today['QN'].N= current_N
        self.alpha_iterator()
        return a_star
        
    def const_policy(self, yesterday, today):  # default policy
        # max_dose = self.policy_params['max_dose']    # Use this to scale dose 
        # max_cohort= self.policy_params['max_cohort']
        dose = today['cohort'] * self.max_dose /  self.max_cohort
        return dose

    def standard_of_care_policy(self, yesterday, today):  # default policy
        # depends on: today's severity
        random_dose = self.my_rng.uniform( low=0.0, high=self.max_dose )
        severity_dependent_dose = random_dose * today['severity'] / self.SEVERITY_CEILING
        return math.floor(10 * severity_dependent_dose)/10 # rounded down to the nearest tenth

    def completely_random_policy(self, yesterday, today):
        # does not depend on anything
        dose = math.floor(10 * self.my_rng.uniform( low=0.0, high=self.max_dose ))/10
        return dose
    
    def linear_change_policy(self, yesterday, today):
        'vary the dose with the number of days. '
        dose = min(self.max_dose, 
                   max(0, 
                       self.const_dose + self.daily_change * (today['day_number'] - self.mid_day)))
        return dose
    
    def linear_severity_policy(self, yesterday, today):
        'vary the dose with the severity. '
        # const_dose is a fixed level independent of severity
        # Daily change is the severity multiplier
        dose = min(self.max_dose, 
                   max(0, 
                       self.const_dose + self.daily_change * (today['severity'] - self.mid_day)))
        return dose
    
    def linear_cum_dose_policy(self, yesterday, today):
        'vary the dose with the severity. '
        # const_dose is a fixed level independent of severity
        # Daily change is the severity multiplier
        dose = min(self.max_dose, 
                   max(0, 
                       self.const_dose + self.daily_change * (yesterday['cum_drug'] - self.mid_day)))
        return dose
    
    def linear_efficacy_policy(self, yesterday, today):
        'vary the dose with the severity. '
        # const_dose is a fixed level independent of severity
        # Daily change is the severity multiplier
        dose = min(self.max_dose, 
                   max(0, 
                       self.const_dose + self.daily_change * (yesterday['efficacy'] - self.mid_day)))
        return dose
        
    def dose_cohort_policy(self, yesterday, today):
        cohort = today['cohort']
        # print(f'c {cohort}')
        dose = self.max_dose * cohort / (self.num_cohorts - 1)
        return dose
    
if __name__ == '__main__':
    params.update({'max_dose': 1.2, 'max_cohort': 2})
    x = BogoPolicies(**params)
    today = {'cohort':1}
    print('dose:', x.const_policy(None, today))
