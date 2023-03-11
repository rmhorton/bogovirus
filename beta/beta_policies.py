# beta_policies.py
#
# 13 March 2023

import os, re, sys
import math

def const_policy(yesterday, today):  # default policy
    dose = CONST_DOSE
    return dose

def standard_of_care_policy(yesterday, today):  # default policy
    # depends on: today's severity
    random_dose = my_rng.uniform( low=0.0, high=MAX_DOSE )
    severity_dependent_dose = random_dose * today['severity'] / SEVERITY_CEILING
    return math.floor(10 * severity_dependent_dose)/10 # rounded down to the nearest tenth

def completely_random_policy(yesterday, today):
    # does not depend on anything
    dose = math.floor(10 * my_rng.uniform( low=0.0, high=MAX_DOSE ))/10
    return dose


def dose_cohort_policy(yesterday, today):
    cohort = today['cohort']
    dose = MAX_DOSE * cohort / (NUM_COHORTS - 1)
    return dose
