# simulation.py
# bogovirus simulation, from DB notebook "Bogovirus_simulation"
# Note this file is not used since this code has ben incorporated in the module benvs/online. 
# 6 March 2023  JMA

import os, re, sys
import math
import pandas as pd
import numpy as np
from numpy.random import default_rng
import datetime as dt
# import matplotlib.pyplot as plt
# import seaborn as sn

sys.path.append('beta\benvs')
from policies import BogoPolicies

my_rng = default_rng(seed=None)

NUM_COHORTS = 1 
MAX_DOSE = 3.0  # we want doses from 0 to 1.5 in 0.1 increments
SEVERITY_CEILING = 125; # Max expected severity.
MAX_DAYS = 100 
SAMPLES = 10

def new_patient(patient_id, day_number):
    'Return a dict of a patient with an initial infection and its severity.'
    return {
            'patient_id': patient_id,             # None of these variable are part of the state
            'cohort': patient_id % NUM_COHORTS,   # 
            'day_number': day_number,             # The stage
            # e.g. An array of length 1, of random integers, 20 <= rv < 40
            'infection': my_rng.integers(low=20, high=40, size=1)[0],
            'severity': my_rng.integers(low=10, high=30, size=1)[0],
            'cum_drug' : 0,
            'outcome':'none',
            'efficacy': 0,
            'drug': 0,
           }


def get_infection(yesterday):
    # depends on: infection_prev
    progression = my_rng.integers(low=0, high=10, size=1)[0]
    return yesterday['infection'] + progression

# def get_drug(row):
#     'Treatment policy...'
#     # depends on: severity
#     # treatment threshold for today
#     drug_threshold = my_rng.integers(low=10, high=50, size=1)[0] 
#     if NUM_COHORTS > 1: # Dose depends on cohort
#         cohort = row['patient_id'] % NUM_COHORTS
#         dose = MAX_DOSE * cohort / (NUM_COHORTS - 1)      
#     else:
#         dose = math.floor(10 * my_rng.uniform( low=0.0, high=MAX_DOSE ))/10  # MAX_DOSE+0.1
#     drug = dose if row['severity'] > drug_threshold else 0
#     return drug

def get_severity(yesterday):
    # depends on: yesterday's severity, infection, & efficacy
    noise = my_rng.normal(loc=0, scale=1, size=1)
    severity_next = yesterday['severity'] * 1.1 + \
                    yesterday['infection'] * 0.1 - \
                    yesterday['efficacy'] + \
                    noise
    return severity_next
    
def get_cum_drug(yesterday, today, proportion=0.7):
    'Mix the current drug level and the current dose in fixed proportion.'
    # depends on: cum_drug_prev, drug
    noise = my_rng.normal(loc=0, scale=0.01, size=1)  # !!! surprisingly sensitive !!!
    r = proportion + noise  # # larger value responds more slowly to changes in dose
    return yesterday['cum_drug'] * r + today['drug'] * (1 - r)

def standard_of_care_policy(yesterday, today):  # default policy
    # depends on: today's severity
    random_dose = my_rng.uniform( low=0.0, high=MAX_DOSE )
    severity_dependent_dose = random_dose * today['severity'] / SEVERITY_CEILING
    return math.floor(10 * severity_dependent_dose)/10 # rounded down to the nearest tenth


def get_efficacy(today):
    # depends on today's drug and cum_drug
    # The amount by which severity will be reduced. 
    # Maybe this shold be a proportion not a fixed amount? Severity can be negative this way.
    noise = my_rng.normal(loc=0, scale=1, size=1)
    sigmoid = lambda x: 1 - 1/(1 + math.exp(-(x - 0.6)/0.1))  # starts at 1, goes to zero
    efficacy = 12 * today['drug'] * sigmoid( today['cum_drug'] ) + noise
    return efficacy

# def get_toxicity(row):
#     # depends on cum_drug
#     return row['cum_drug'] ** 6 

def get_outcome(today):
  # depends on today's severity, infection, and cum_drug
  # possible outcomes: die, recover, none
  noise = my_rng.normal(loc=0, scale=0.1, size=1)
  mortality_threshold = 1.0 + noise # my_rng.uniform(low=0.9, high=1.1)
  if (today['severity']/SEVERITY_CEILING > mortality_threshold):
    return 'die'
  elif today['infection'] >= 100:
    return 'recover'
  else:
    return 'none'

def cycle(patient_id, day_number, yesterday, policy):
    today = {} # new row
    if yesterday is None:
        today = new_patient(patient_id, day_number)
    else:
        today = {
            'patient_id': yesterday['patient_id'],
            'cohort': yesterday['cohort'],
            'day_number': day_number,
        }
        # order matters here
        today['infection'] = get_infection(yesterday)
        today['severity']  = get_severity(yesterday)
        today['drug']      = policy(yesterday, today)
        today['cum_drug']  = get_cum_drug(yesterday, today)
        today['efficacy']  = get_efficacy(today)
        today['outcome']   = get_outcome(today)
    
    return today

def sim_patient(patient_id, policy):
    max_days = MAX_DAYS

    patient_results = []
    yesterday = None
    for day_number in range(max_days):
        today = cycle(patient_id=patient_id, day_number=day_number, yesterday=yesterday, policy=policy)
        patient_results.append(today)
        if today['outcome'] != 'none':
            break
        yesterday = today

    patient_pdf = pd.DataFrame(patient_results)
    return patient_pdf


# TODO - do this with the env not here. 
def sim_population(num_patients, policy=standard_of_care_policy):
    'Run a simulation episode '
    patient_list = []
    for patient_id in range(num_patients):
        patient_list.append(sim_patient(patient_id, policy))
    patient_df = pd.concat(patient_list)
    return patient_df

# def pickle_df(the_df):
#     'Save the simulation as a dataset.'
#     # Return a datetime string suitable as part of a filename
#     x = str(dt.datetime.now())
#     dt_str = re.sub(r'\..*$', '', x.replace(' ','_').replace(':', '-'))
#     the_df.to_pickle('patient_data'+dt_str+'.pkl')


if __name__ == '__main__':
    # test running the simulation here
    patient_data = sim_population(NUM_COHORTS * SAMPLES)

    num_recovered = np.sum(patient_data.outcome == 'recover')
    num_died = np.sum(patient_data.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died")

    # pickle_df(patient_data)