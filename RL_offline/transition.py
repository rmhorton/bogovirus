# transition.py
# Simulate Markcov transition matrici using the formulae in simulation.py
# JMA 16 Sept 2022
# See https://github.com/nelsonuhan/stochasticdp

# from audioop import maxpp
import os, re, sys
import pandas as pd
import numpy as np
from numpy.random import default_rng
import datetime as dt

import simulation as sm 

DBG = True
DISCRETIZATIONS = 10  # No of bins for each discrete var.
INFECTION_DAYS = 20  # Number of days expected for the disease to run it's course 
# get MAX_DAYS from sm.MAX_DAYS
SEED = 42
NS = None

my_rng = default_rng(SEED)

def infection_transition(L):
    '''Simulate the infection transition.
    Live thru the infection to this level, and you recover. 
    L  = rv upper limit'''
    # For discrete state m, the 
    # P of transition to m+1 is constant, so the increment depends on 
    # the number of discretizations, n, and the number of steps, T.
    # At each step, that is each stage transition, the expected days
    # increments by one. Since the number of discrete states is less
    # than the number of expected days, the P of transition is 
    transition_fraction = (INFECTION_DAYS - DISCRETIZATIONS) / INFECTION_DAYS
    # We need to set the transition probability from m to m+1 equal
    # to this. Using a uniform distribution over number of days, for
    # a maximum day-step of L, the probability of transition is then
    # Solve for s: 
    # s = L (T-n)/T
    transition_threshold = L * transition_fraction
    progress = my_rng.integers(low=0, high=L, size=1)[0]
    return progress > transition_threshold

def monotonic_growth_matrix():
    '''A markov matrix with for expected transitions over n states  in T steps'''
    transition_P = (INFECTION_DAYS - DISCRETIZATIONS) / INFECTION_DAYS
    complement_P = 1 - transition_P
    f = [transition_P, complement_P]
    f.extend([0] * (DISCRETIZATIONS -1))
    g = (DISCRETIZATIONS-1) * f
    g.append(1.0)
    return np.array(g).reshape((DISCRETIZATIONS, DISCRETIZATIONS))

### Build matrici from the simulation. 

def read_simulation(the_file = 'patient_data2022-09-12_15-13-57.pkl'):
    'Use the random simulation data to learn the CPTs '
    the_df = pd.read_pickle(the_file)
    the_df = the_df[['infection_prev', 'severity_now',
       'cum_Tx_prev', 'infection_now', 'Tx', 'cum_Tx', 'severity_next', 'die',
       'recover']]
    return the_df

def discrete_series(the_range, the_series, prf):
    global NS
    cat_df = \
        pd.cut(the_series, \
         np.linspace(the_range[0], the_range[1], DISCRETIZATIONS+1),
        labels =  [f'{prf}{n}' for n in range(DISCRETIZATIONS)],
        include_lowest=True)
    if DBG:
        new_series = pd.concat([the_series, cat_df], axis=1)
        print(new_series.groupby(level=0).size())
        NS = new_series
    return cat_df 

def discretize_simulation(the_df):
    cat_df = pd.DataFrame()
    ## infection
    mins = dict(the_df.apply(min))
    maxs = dict(the_df.apply(max))
    inf_range = (min(mins['infection_prev'], mins['infection_now']), 
    max(maxs['infection_prev'], maxs['infection_now']))
    cat_df['infection_prev'] = discrete_series(inf_range, the_df['infection_prev'], 'ip')
    cat_df['infection_now'] = discrete_series(inf_range, the_df['infection_now'], 'in')
    ## severity
    inf_range = (min(mins['severity_next'], mins['severity_now']), 
    max(maxs['severity_next'], maxs['severity_now']))
    cat_df['severity_next'] = discrete_series(inf_range, the_df['severity_next'], 'st')
    cat_df['severity_now'] = discrete_series(inf_range, the_df['severity_now'], 'sn')
    ## Tx
    inf_range = (min(mins['cum_Tx_prev'], mins['cum_Tx'], mins['Tx']), 
    max(maxs['cum_Tx_prev'], maxs['cum_Tx'],  maxs['Tx']))
    cat_df['cum_Tx_prev'] = discrete_series(inf_range, the_df['cum_Tx_prev'], 'cp')
    cat_df['cum_Tx'] = discrete_series(inf_range, the_df['cum_Tx'], 'cn')
    cat_df['Tx'] = discrete_series(inf_range, the_df['Tx'], 'ct')
    ## outcomes are T,F -- convert to recover = 1, die = -1
    cat_df['outcome'] = the_df['recover'].apply(int) - the_df['die'].apply(int)
    return cat_df


if __name__ == '__main__':
#     x = np.transpose(np.array([1.0] + (DISCRETIZATIONS-1) * [0] ))
# for n in range(40):
#     x = x @ monotonic_growth_matrix() 
#     print(n, x)
    df = read_simulation() #'patient_data_random_dose1000.parquet')
    cat_df = discretize_simulation(df)
    cn_cpt = pd.crosstab(cat_df['cum_Tx'], [cat_df['cum_Tx_prev'], cat_df['Tx']])
    in_cpt = pd.crosstab(cat_df['infection_now'], cat_df['infection_prev'], )
    sn_cpt = pd.crosstab(cat_df['severity_next'], [cat_df['severity_now'], cat_df['infection_now'], cat_df['cum_Tx']])
    print(cat_df.head())
