# simulation.py
# bogovirus simulation, from DB notebook "Bogovirus_simulation"
# 9 Sept 2022  JMA

import os, re, sys
import math
import pandas as pd
import numpy as np
from numpy.random import default_rng
import datetime as dt
# import matplotlib.pyplot as plt
# import seaborn as sn

my_rng = default_rng(seed=None)

NUM_COHORTS = 1 
MAX_DOSE = 1.5  # we want doses from 0 to 1.5 in 0.1 increments
MAX_DAYS = 100 
SAMPLES = 10

def new_patient(patient_id, day_number):
    'Return a dict of a patient with an initial infection and its severity.'
    return {
            'patient_id': patient_id,             # None of these variable are part of the state
            'cohort': patient_id % NUM_COHORTS,   # 
            'day_number': day_number,             # The stage
            # e.g. An array of length 1, of random integers, 20 <= rv < 40
            'infection_prev': my_rng.integers(low=20, high=40, size=1)[0],
            'severity': my_rng.integers(low=10, high=30, size=1)[0],
            'cum_drug_prev' : 0
           }


def get_infection(row):
    'Increment an infection value randomly by an amount 0 to 9.'
    # depends on: infection_prev
    progression = my_rng.integers(low=0, high=10, size=1)[0]
    return row['infection_prev'] + progression

def get_drug(row):
    'Treatment policy...'
    # depends on: severity
    # treatment threshold for today
    drug_threshold = my_rng.integers(low=10, high=50, size=1)[0] 
    if NUM_COHORTS > 1: # Dose depends on cohort
        cohort = row['patient_id'] % NUM_COHORTS
        dose = MAX_DOSE * cohort / (NUM_COHORTS - 1)      
    else:
        dose = math.floor(10 * my_rng.uniform( low=0.0, high=MAX_DOSE ))/10  # MAX_DOSE+0.1
    drug = dose if row['severity'] > drug_threshold else 0
    return drug
    
def get_cum_drug(row, proportion=0.6):
    'Mix the current drug level and the current dose in fixed proportion.'
    # depends on: cum_drug_prev, drug
    r = proportion  # # larger value responds more slowly to changes in dose
    return row['cum_drug_prev'] * r + row['drug'] * (1 - r)


def get_severity_next(row):
    'Severity transition ... '
    # depends on: severity, infection_now, drug, cum_drug
                    # increase due to current severity
    severity_next = (row['severity'] * 1.1 + 
                    # increase due to current infection
                    row['infection'] * 0.1 - 
                    # decrease from current dose as a fraction of the cumulative dose
                    20 * row['drug']/(row['cum_drug'] + 1))  # larger coefficient makes drug work better
    return severity_next  # Check - should return a number. 
    
def get_toxicity(row):
    # depends on cum_drug
    return row['cum_drug'] ** 6 


def get_outcome(row):
  # depends on severity, infection, and toxicity
  # possible outcomes: die, recover, none
  mortality_threshold = 1.0 # my_rng.uniform(low=0.9, high=1.1)
  if (row['toxicity'] + row['severity']/125) > mortality_threshold:
    return 'die'
  elif row['infection'] >= 100:
    return 'recover'
  else:
    return 'none'


def cycle(patient_id, day_number, prev_row):
    'Advance a patient history one stage forward.'
    if prev_row is None:
        row = new_patient(patient_id, day_number)
    else:
        row = {
                'patient_id': prev_row['patient_id'],
                'day_number': day_number,
                'infection_prev': prev_row['infection'],
                'severity': prev_row['severity_next'],
                'cum_drug_prev' : prev_row['cum_drug']
        }
        
    row['infection'] = get_infection(row)
    row['drug'] = get_drug(row)
    row['cum_drug'] = get_cum_drug(row)
    row['severity_next'] = get_severity_next(row)
    row['toxicity'] = get_toxicity(row)
    row['outcome'] = get_outcome(row)
    return row


def sim_patient(patient_id):
    'cycle the patient pop through a sequence of days'
    max_days = MAX_DAYS
    patient_results = []
    old_row = None
    for day_number in range(max_days):
        new_row = cycle(patient_id=patient_id, day_number=day_number, prev_row=old_row)
        patient_results.append(new_row)
        if new_row['outcome'] != 'none':
            break
        old_row = new_row
    patient_pdf = pd.DataFrame(patient_results)
    return patient_pdf


def sim_population(num_patients):
    'Run a simulation episode '
    patient_list = []
    for patient_id in range(num_patients):
        patient_list.append(sim_patient(patient_id))

    patient_df = pd.concat(patient_list)
    return patient_df

def pickle_df(the_df):
    'Save the simulation as a dataset.'
    # Return a datetime string suitable as part of a filename
    x = str(dt.datetime.now())
    dt_str = re.sub(r'\..*$', '', x.replace(' ','_').replace(':', '-'))
    the_df.to_pickle('patient_data'+dt_str+'.pkl')


if __name__ == '__main__':

    patient_data = sim_population(NUM_COHORTS * SAMPLES)

    num_recovered = np.sum(patient_data.outcome == 'recover')
    num_died = np.sum(patient_data.outcome == 'die')
    print(f"{num_recovered} patients recovered and {num_died} died")

    pickle_df(patient_data)