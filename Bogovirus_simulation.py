# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Human Bogovirus Syndrome Simulator
# MAGIC 
# MAGIC 
# MAGIC Human Bogovirus Syndrome (HBS) is a fictitious disease, caused by a newly discovered virus _Bogovirus_.
# MAGIC 
# MAGIC Infection can be established by detecting the Bogovirus Coat Protein (BCP) in blood samples. Infection leads to a problematic disorder of the gizzard, which is generally fatal if untreated. Survival depends on managing the disorder until the disease runs its course, which usually takes about 3 weeks assuming the patient does not die.
# MAGIC 
# MAGIC Response to treatment depends on the infection state, the current severity of the disorder, administration of the treatment, and the cumulative dose of the treatment.
# MAGIC 
# MAGIC The effectiveness of the treatment in reducing severity of the disorder decreases as the cumulative dose increases. In addition, high clumulative doses are toxic. The cumulative dose of treatment is modeled by exponential decay. 
# MAGIC 
# MAGIC ![HBS diagram](https://rmhorton.github.io/bogovirus/HBS_diagram.png)
# MAGIC 
# MAGIC This sequence diagram expands out the influences across time cycles:
# MAGIC 
# MAGIC ![HBS sequence](https://rmhorton.github.io/bogovirus/HBS_sequence.png)
# MAGIC 
# MAGIC Variables may depend on either the previous cycle or the next cycle, but not both. Variables and influences included in the model are shown with solid strokes, and those not in the model are dotted.
# MAGIC 
# MAGIC * `infection`: records how far the patient has passed through the course of the infection (you can think of this as percent progress; once it passes 100 and you have not died, you recover).
# MAGIC * `drug`: a dose of treatment. These are unit doses for now (you either get the standard dose or not), but we could make this an adjustable quantity.
# MAGIC * `cum_drug`: The accumulated dose of the treatment drug. This is an exponential average, and is subject to decay if treatment doses are not administered.
# MAGIC * `severity`: quantified level of severity of the disorder caused by the infection. The worse this gets, the more likely you are to die.
# MAGIC * `die/recover`: if severity gets high enough, your chances of death increase. If the infection runs its course and you don’t die, you recover.
# MAGIC * `toxicity`: a function of `cum_drug` that is much more pronounced at high cumulative dose.
# MAGIC 
# MAGIC 
# MAGIC [Powerpoint file](https://microsoft-my.sharepoint.com/:p:/p/joagosta/EXa8N5FFBEtOps31Qfs9YYgBmbolFD0mcVVS3xmw574oLA?e=7xGghb) with the editable versions of these diagrams.
# MAGIC 
# MAGIC ## To Do
# MAGIC 
# MAGIC The purpose of this simulation is to generate data suitable for building causal models. We want the causal models to capture relationships in the data in a way that allows us to try 'what-if' scenarios. Examples include:
# MAGIC * What is the optimal dose?
# MAGIC * Would a different dosing protocol work better?
# MAGIC 
# MAGIC These questions are most interesting if there is some optimal answer (a Goldilocks problem). A simple way to avoid just giving everybody the treatment as early as possible might be to have some toxicity for the treatment, and maybe have a higher fraction of people who would survive without treatment. Maybe have a constant rate of degradation, so that maintaining a low cumulative dose would be sustainable, but a high cumulaive dose would not be. It would be good if there were an optimal dose and schedule.

# COMMAND ----------

import pandas as pd
import numpy as np
import math
from numpy.random import default_rng

my_rng = default_rng(seed=42)

NUM_COHORTS = 16
MAX_DOSE = 1.5  # we want doses from 0 to 1.5 in 0.1 increments

def new_patient(patient_id, day_number):
    return {
            'patient_id': patient_id,
            'cohort': patient_id % NUM_COHORTS,
            'day_number': day_number,
            'infection_prev': my_rng.integers(low=20, high=40, size=1)[0],
            'severity': my_rng.integers(low=10, high=30, size=1)[0],
            'cum_drug_prev' : 0
           }


def get_infection(row):
    # depends on: infection_prev
    progression = my_rng.integers(low=0, high=10, size=1)[0]
    return row['infection_prev'] + progression


def get_drug(row):
    # depends on: severity
    # treatment threshold for today
    tx_threshold = my_rng.integers(low=10, high=50, size=1)[0] 
    if NUM_COHORTS > 1: # dose depends on cohort
      cohort = row['patient_id'] % NUM_COHORTS
      dose = MAX_DOSE * cohort / (NUM_COHORTS - 1)
    else: # random dose
      dose = math.floor(10 * my_rng.uniform( low=0.0, high=MAX_DOSE ))/10  # MAX_DOSE+0.1
    drug = dose if row['severity'] > tx_threshold else 0
    return drug
    
    
def get_cum_drug(row):
    # depends on: cum_drug_prev, drug
    r = 0.6  # larger value responds more slowly to changes in dose
    return row['cum_drug_prev'] * r + row['drug'] * (1 - r)


def get_severity_next(row):
    # depends on: severity, infection, drug, cum_drug
    severity_next = row['severity'] * 1.1 + \
                    row['infection'] * 0.1 - \
                    20 * row['drug']/(row['cum_drug'] + 1)  # larger coefficient here makes drug work better
    return severity_next
    

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
    if prev_row is None:
        row = new_patient(patient_id, day_number)
    else:
        row = {
            'patient_id': prev_row['patient_id'],
            'cohort': prev_row['cohort'],
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
    max_days = 100

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
    patient_list = []
    for patient_id in range(num_patients):
        patient_list.append(sim_patient(patient_id))

    patient_df = pd.concat(patient_list)
    return patient_df

# COMMAND ----------

# DBTITLE 1,Generate dose-cohort data
# MAGIC %python
# MAGIC 
# MAGIC patient_data = sim_population(NUM_COHORTS * 1000)
# MAGIC 
# MAGIC num_recovered = np.sum(patient_data.outcome == 'recover')
# MAGIC num_died = np.sum(patient_data.outcome == 'die')
# MAGIC print(f"{num_recovered} patients recovered and {num_died} died")

# COMMAND ----------

# DBTITLE 1,Show data for one patient
# last_day[['day_number', 'survive']].groupby('day_number').mean()

# last_day.loc[(last_day.day_number == 8) & (last_day.outcome == 'recover')]

patient_data[patient_data.patient_id == 4456] 

# COMMAND ----------

# What was the assigned dose for each cohort?
cohort_dose = patient_data[ ['cohort', 'drug'] ].groupby('cohort').max()

# What fraction of each cohort died?
last_day = patient_data.groupby('patient_id').last()
last_day['survive'] = [1 if x == 'recover' else 0 for x in last_day['outcome']]
cohort_survive = last_day[ ['cohort', 'survive'] ].groupby('cohort').mean()

dose_outcome_pdf = pd.concat([cohort_dose, cohort_survive], axis=1)

print("Best survival fraction:", np.max(dose_outcome_pdf.survive))

import matplotlib.pyplot as plt

ax = dose_outcome_pdf.plot(x='drug', y='survive', 
                      # xlabel='Dose of drug', ylabel='Fraction of cohort surviving', 
                      title="Optimizing dose", legend=False, 
                      figsize=(12, 8), fontsize=18, lw=6)
ax.title.set_size(36)
ax.set_xlabel('Dose of drug',fontdict={'fontsize':24})
ax.set_ylabel('Fraction of cohort surviving',fontdict={'fontsize':24})

best_dose = dose_outcome_pdf.drug[np.argmax(dose_outcome_pdf.survive)]
plt.axvline(x=best_dose, linestyle='dashed', color='red')

# COMMAND ----------

dose_outcome_pdf

# COMMAND ----------

from collections import Counter

Counter(patient_data['cohort']) # Patients who die have fewer days in the hospital!

# COMMAND ----------

# create database if not exists bogovirus
spark.sql("use bogovirus")
spark.createDataFrame(patient_data).write.mode("overwrite").saveAsTable("patient_data_cohort_dose")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC library(dplyr)
# MAGIC library(sparklyr)
# MAGIC library(ggplot2)
# MAGIC 
# MAGIC sc <- spark_connect(method='databricks')
# MAGIC 
# MAGIC sdf_sql(sc, "select * from bogovirus.patient_data_cohort_dose") %>% 
# MAGIC   group_by(patient_id) %>% filter(day_number==max(day_number)) %>% 
# MAGIC   collect %>%  # bring to R to compute mean
# MAGIC   group_by(cohort) %>% summarize(mean_survival = mean(outcome != 'die')) %>% 
# MAGIC   mutate(dose=as.numeric(cohort)/10, survival=mean_survival) %>% select(dose, survival) %>% as.data.frame

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC library(dplyr)
# MAGIC library(sparklyr)
# MAGIC library(ggplot2)
# MAGIC 
# MAGIC sc <- spark_connect(method='databricks')
# MAGIC 
# MAGIC sdf_sql(sc, "select * from bogovirus.patient_data_cohort_dose") %>% 
# MAGIC   group_by(patient_id) %>% filter(day_number==max(day_number)) %>% 
# MAGIC   collect %>%  # bring to R to compute mean
# MAGIC   group_by(cohort) %>% summarize(mean_survival = mean(outcome != 'die')) %>% 
# MAGIC   ggplot(aes(x=cohort, y=mean_survival)) + geom_line()

# COMMAND ----------

# DBTITLE 1,Generate random dose dataset
NUM_COHORTS = 1

patient_data_random_dose = sim_population(500000)

spark.createDataFrame(patient_data_random_dose).write.mode("overwrite").saveAsTable("bogovirus.patient_data_random_dose")

# COMMAND ----------

# MAGIC %sql
# MAGIC use bogovirus;
# MAGIC 
# MAGIC select count(*) from patient_data_random_dose

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Simulation without toxicity

# COMMAND ----------

NUM_COHORTS = 15

def get_toxicity(row):
    # non-toxic version!!
    return 0

notox_pop = sim_population(1000 * NUM_COHORTS)

# COMMAND ----------

# What was the assigned dose for each cohort?
cohort_dose = notox_pop[ ['cohort', 'drug'] ].groupby('cohort').max()
cohort_dose

# COMMAND ----------


 
# What fraction of each cohort died?
last_day = notox_pop.groupby('patient_id').last()
last_day['survive'] = [1 if x == 'recover' else 0 for x in last_day['outcome']]
cohort_survive = last_day[ ['cohort', 'survive'] ].groupby('cohort').mean()
 
dose_outcome_pdf = pd.concat([cohort_dose, cohort_survive], axis=1)
 
print("Best survival fraction:", np.max(dose_outcome_pdf.survive))
 
ax = dose_outcome_pdf.plot(x='drug', y='survive', 
                      # xlabel='Dose of drug', ylabel='Fraction of cohort surviving', 
                      title="Optimizing dose", legend=False, 
                      figsize=(12, 8), fontsize=18, lw=6)
ax.title.set_size(36)
ax.set_xlabel('Dose of drug',fontdict={'fontsize':24})
ax.set_ylabel('Fraction of cohort surviving',fontdict={'fontsize':24})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## "No-toxicity" data for Maryam

# COMMAND ----------

NUM_COHORTS = 1

notox_pop_random = sim_population(10000)

spark.createDataFrame(notox_pop_random).write.mode("overwrite").saveAsTable("bogovirus.notox_pop_random")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from bogovirus.notox_pop_random

# COMMAND ----------


