# gym environment for the bogovirus using the true simulator 
#  JMA 26 sept 2022

import os, re, sys
import numpy as np
import pandas as pd
# from numpy.random import default_rng
import gym
from gym.spaces import Box, Dict
import mlflow

import simulation as sm

SEED = None       # Use a random seed each run

class BogoEnv(gym.Env):
    'An environment class derived from the gym environment'

    def __init__(self ) -> None:
        'Call this once, and reuse it for all episodes'
        # self.n_neighbors = N_NEIGHBORS 
        # mlflow.log_param('Neighbors', N_NEIGHBORS)    # This should be done in the calling app, not the env. 
        self.render_mode = None
        # fix the type warnings for these
        self.action_space = Dict(
            {"Dose": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)}
        )
        self.observation_space = Dict(
            {"Infection": Box(low=0, high=150, shape=(1,), dtype=np.float32), 
             "Severity": Box(low=0, high=170, shape=(1,), dtype=np.float32),
             "CumulativeDrug": Box(low=0, high=1, shape=(1,), dtype=np.float32)}
        )
        # Let the RL set the action -- do not assign patients to cohorts
        # State variables.
        self.Infection = None
        self.Severity = None
        self.CumulativeDrug = None
        self.stage = 0               # Not a state variable, but the sim tracks it
        # Immediate rewards
        self.one_day = -1
        self.recover = 100
        self.die = -200

    def reset(self, *, sd=SEED, options=None) -> dict:
        'Initialize an episode'
        super().reset(seed=sd)  # TODO why doesn't this work? 
        # self.my_rng = default_rng(seed=SEED) simulation.py manages this. 
        # State variables.
        # THe state is observable, so we use observation as the state. 
        # Of course for a constant policy observability is moot. 
        row = sm.new_patient(patient_id=0, day_number=self.stage)
        self.Infection = row['infection_prev']
        self.Severity = row['severity']           # Severity is "now" not "prev"
        self.CumulativeDrug = row['cum_drug_prev']
        #  Features for the predictor -- representing the current state. Only those features samples will be searched on. 
        info = {'stage': self.stage}   # Just a place to return additional info, e.g. for diagnostics
        return self._get_ob(), info


    def step(self,  actions):
        'Increment the state at each stage in an episode, and terminate on death or recovery.'
        terminated = False # reached a final state - dead or recovered
        reward = self.one_day
        # Dont call cycle.  Use some of it's contents
        row = {
            'patient_id': 0,
            'day_number': self.stage +1,
            'infection_prev': self.Infection,
            'severity': self.Severity,
            'cum_drug_prev' : self.CumulativeDrug 
        }
        # Note, the order these are called matters.
        row['drug'] = actions
        row['infection'] = sm.get_infection(row)
        row['cum_drug'] = sm.get_cum_drug(row)
        row['severity_next'] = sm.get_severity_next(row)
        row['toxicity'] = sm.get_toxicity(row)
        row['outcome'] = sm.get_outcome(row)
        # Die if the drug concentration or severity is high
        if row['outcome'] =='die' :
            reward = self.die
            terminated = True
        # If the patient didn't die, but outlasted the infection, recover. 
        elif row['outcome'] =='recover' :
            reward = self.recover
            terminated = True
        self.Infection = row['infection']
        self.Severity = row['severity_next']
        self.CumulativeDrug = row['cum_drug']
        info = {"stage": self.stage}
        return self._get_ob(), reward, terminated, False , info

    def _get_ob(self):
        'A convenience function to format the state output '
        return {"Infection":self.Infection, "Severity":self.Severity, "CumulativeDrug":self.CumulativeDrug}
