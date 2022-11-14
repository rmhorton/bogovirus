# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC This simulation trained on simulated data ('simsim') uses the `gRain` package. While this package is available through CRAN, it depends on several packages from Bioconductor. Here I install the package and dependencies, which I do as [notebook-scoped libraries](https://docs.databricks.com/libraries/notebooks-r-libraries.html).

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC use bogovirus;
# MAGIC 
# MAGIC select drug, count(*) tally from patient_data_random_dose group by drug order by drug

# COMMAND ----------

# DBTITLE 1,Generate categorized versions of the simulated data
# MAGIC %sql
# MAGIC 
# MAGIC use bogovirus;
# MAGIC 
# MAGIC refresh patient_data_random_dose;
# MAGIC 
# MAGIC -- (Mostly) 10 categories per variable
# MAGIC create or replace temporary view patient_data_random_dose_10cat as (
# MAGIC   with pdrd as (
# MAGIC     select * 
# MAGIC       , floor(10*percent_rank(toxicity) OVER (ORDER BY toxicity))/10 as toxicity_pct
# MAGIC       , drug/(cum_drug+1) as efficacy
# MAGIC       from patient_data_random_dose
# MAGIC   )
# MAGIC   select patient_id
# MAGIC     , day_number
# MAGIC     , case when cum_drug_prev >= 0.95 then 95 else floor(cum_drug_prev*10)*10 end as cum_drug_prev
# MAGIC     , case when cum_drug >= 0.95 then 95 else floor(cum_drug*10)*10 end as cum_drug
# MAGIC     , case when infection_prev >= 99 then 99 else floor(infection_prev/10)*10 end as infection_prev
# MAGIC     , case when infection >= 99 then 99 else floor(infection/10)*10 end as infection
# MAGIC     , case when severity < 0 then 0 when severity > 120 then 12 else floor(severity/10) end as severity
# MAGIC     , case when severity_next < 0 then 0 when severity_next > 120 then 12 else floor(severity_next/10) end as severity_next
# MAGIC     , cast(int(drug * 10) as string) as drug
# MAGIC     , outcome
# MAGIC     , floor(10*efficacy) as efficacy                                                                              -- ***
# MAGIC     , case when infection >= 99 then 1 else 0 end as infection_over                                               -- ***
# MAGIC     , case when cum_drug >= 0.95 then 'high' when cum_drug >= 0.90 then 'medium' else 'low' end as toxicity_group -- ***
# MAGIC     from pdrd
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from patient_data_random_dose_10cat

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Fit Bayes Net in gRain

# COMMAND ----------

# DBTITLE 1,Install packages (~8.5 minutes)
suppressMessages({  # suppressPackageStartupMessages
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", quiet = TRUE)
  BiocManager::install("graph")
  BiocManager::install("Rgraphviz")
  BiocManager::install("RBGL")
  install.packages('gRbase', quiet = TRUE)
  install.packages('gRain', quiet = TRUE)
})

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from patient_data_random_dose_10cat

# COMMAND ----------

library(dplyr)
library(sparklyr)

sc <- spark_connect(method='databricks')

patient_data <- sdf_sql(sc, "select * from patient_data_random_dose_10cat") %>% collect

not_categorical <- c("patient_id", "day_number")
cat_features <- setdiff(names(patient_data), not_categorical)

# how many examples of each category?
patient_data[cat_features] %>% lapply(table)

# COMMAND ----------

patient_data %>% sapply(class)

# COMMAND ----------

exclude <- c("patient_id", "day_number")
char_cols <- c("outcome", "toxicity_group")
numeric_features <- setdiff(names(patient_data), c(exclude, char_cols))
features <- c(numeric_features, "toxicity_group")
keepers <- c(features, "outcome")

for (col in char_cols){
  print(sprintf("Converting column '%s' to factor", col))
  patient_data[[col]] <- factor(patient_data[[col]])
}

for (col in numeric_features){
  print(sprintf("Converting column '%s' to factor", col))
  patient_data[[col]] <- factor(sprintf('c%02d', as.numeric(patient_data[[col]])))
}


# COMMAND ----------

patient_data %>% # head %>% as.data.frame %>% 
sapply(class)

# COMMAND ----------

# patient_data %>% mutate(drug=as.numeric(drug)) %>% ggplot(aes(x=drug)) + geom_density()
table(patient_data$drug)

# COMMAND ----------

library(gRain)

dag.bogo <- dag(~ infection:infection_prev +
                  infection_over:infection +            # ***
                  cum_drug:cum_drug_prev:drug +
                  toxicity_group:cum_drug +             # ***
                  outcome:infection_over:toxicity_group:severity +
                  drug:severity +
                  efficacy:drug:cum_drug +              # ***
                  severity_next:severity:efficacy:infection
)

plot(dag.bogo)

# COMMAND ----------

# smooth to avoid zero entries in the CPTs
SMOOTHING <- 1e-15  # .Machine$double.eps == 2.220446e-16
bn.bogo_learned <- grain(dag.bogo, data=patient_data[keepers], smooth=SMOOTHING)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # The 'SimSim' simulation

# COMMAND ----------

# DBTITLE 1,Adjust the CPTs to extrapolate to more extreme combinations of inputs
### Monotonic extrapolation
# Overwrite regions of uniform probability with the nearest trained value 
# Extrapolate assuming extreme values beyond the range of the dataset equal most extreme value in dataset.

# fill values forward
fill_forward <- function(M){
  sd_threshold <- 0.1
  for (i in 2:ncol(M)){
    if (sd(M[,i]) < sd_threshold)
      M[,i] <- M[,i-1]
  }
  M
}


# fill values backward
fill_backward <- function(M){
  sd_threshold <- 0.1
  for (i in (ncol(M) - 1):1){
    if (sd(M[,i]) < sd_threshold)
      M[,i] <- M[,i+1]
  }
  M
}

NUM_COHORTS <- 15
SAMPLES_PER_COHORT <- 100
PLOT_TITLE <- "Extrapolated Efficacy and Outcome"

cptlist <- bn.bogo_learned$cptlist

# # efficacy ~ cum_drug_cat
drug_categories <- dimnames(cptlist$efficacy)[['drug']]
for (drug_cat in drug_categories){
  M <- cptlist$efficacy[, , drug=drug_cat]
  cptlist$efficacy[, , drug=drug_cat] <- M %>% fill_forward %>% fill_backward
}

# # outcome ~ severity
infection_over_categories <- dimnames(cptlist$outcome)[['infection_over']]
toxicity_group_categories <- dimnames(cptlist$outcome)[['toxicity_group']]
for (infection_over_cat in infection_over_categories){
  for (toxicity_group_cat in toxicity_group_categories){
    M <- cptlist$outcome[, infection_over=infection_over_cat, toxicity_group=toxicity_group_cat, ]
    cptlist$outcome[,
                    infection_over=infection_over_cat,
                    toxicity_group=toxicity_group_cat, ] <- M %>% fill_forward %>% fill_backward
  }
}

modified_cpt <- compileCPT(cptlist)
bn.bogo <- grain(modified_cpt)

# COMMAND ----------

# These functions return a 'row', which is really just a string vector of (mostly) category names.
new_patient <- function(patient_id, day_number){
  # nomally sampling probabilities will come from BayesNet, but in this case they are computed from day 0 data
  # querygrain(bn.bogo, nodes=c("infection_prev"))$infection_prev
  day0 <- patient_data[patient_data$day_number==0, c('infection_prev', 'severity')] %>% 
            lapply(table)
  c(
    'patient_id' = patient_id,
    'day_number' = day_number,
    'infection_prev' = base::sample(names(day0$infection_prev), 1, prob=day0$infection_prev),
    'severity' = base::sample(names(day0$severity), 1, prob=day0$severity),
    'cum_drug_prev' = 'c00'
  )
}

get_infection <- function(row){
  # depends on: infection_prev
  bn.bogo.infection <- setEvidence(bn.bogo, evidence=row)
  P_infection <- querygrain(bn.bogo.infection, nodes=c("infection"))$infection
  row['infection'] <- base::sample(names(P_infection), 1, prob=P_infection)
  row
}


get_infection_over <- function(row){
  # depends on: infection
  bn.bogo.infection <- setEvidence(bn.bogo, evidence=row)
  P_infection_over <- querygrain(bn.bogo.infection, nodes=c("infection_over"))$infection
  row['infection_over'] <- base::sample(names(P_infection_over), 1, prob=P_infection_over)
  row
}


get_drug <- function(row, dose=NA){
  # depends on: severity
  # dose is a category name, like 'c02'
  # capture treatment threshold from whether drug was administered, but use the provided dose
  if (is.na(dose)){
    bn.bogo.drug <- setEvidence(bn.bogo, evidence=row)
    P_drug <- querygrain(bn.bogo.drug, nodes=c("drug"))$drug
    drug = base::sample(names(P_drug), 1, prob=P_drug)
  } else {
    drug = dose
  }
  row['drug'] = drug
  row
}


get_cum_drug <- function(row){
  # depends on: cum_drug_prev, drug
  bn.bogo.cum_drug <- setEvidence(bn.bogo, evidence=row)
  P_cum_drug <- querygrain(bn.bogo.cum_drug, nodes=c("cum_drug"))$cum_drug
  row['cum_drug'] = base::sample(names(P_cum_drug), 1, prob=P_cum_drug)
  row
}


get_efficacy <- function(row){
  # depends on drug, cum_drug
  bn.bogo.efficacy <- setEvidence(bn.bogo, evidence=row)
  P_efficacy <- querygrain(bn.bogo.efficacy, nodes=c("efficacy"))$efficacy
  row['efficacy'] = base::sample(names(P_efficacy), 1, prob=P_efficacy)
  row
}


get_severity_next <- function(row){
  # depends on: severity, infection, efficacy
  bn.bogo.severity_next <- setEvidence(bn.bogo, evidence=row)
  P_severity_next <- querygrain(bn.bogo.severity_next, nodes=c("severity_next"))$severity_next
  row['severity_next'] = base::sample(names(P_severity_next), 1, prob=P_severity_next)
  row
}


get_toxicity_group <- function(row){
  # depends on: cum_drug
  bn.bogo.toxicity_group <- setEvidence(bn.bogo, evidence=row)
  P_toxicity_group <- querygrain(bn.bogo.toxicity_group, nodes=c("toxicity_group"))$toxicity_group
  row['toxicity_group'] = base::sample(names(P_toxicity_group), 1, prob=P_toxicity_group)
  row
}


get_outcome <- function(row){
  # depends on severity, infection, and toxicity
  bn.bogo.outcome <- setEvidence(bn.bogo, evidence=row)
  P_outcome <- querygrain(bn.bogo.outcome, nodes=c("outcome"))$outcome
  row['outcome'] = base::sample(names(P_outcome), 1, prob=P_outcome)
  row
}


cycle <- function(day_number, patient_id, drug_dose, prev_row=NULL){
  if ( is.null(prev_row) ){
      row = new_patient(patient_id, day_number)
  } else {
      row = c(
        patient_id = prev_row[['patient_id']],
        day_number = day_number,
        infection_prev = prev_row[['infection']],
        severity = prev_row[['severity_next']],
        cum_drug_prev = prev_row[['cum_drug']]
      )
  }
  
  row <- row %>% 
    get_infection %>% 
    get_infection_over %>%
    get_drug(dose=drug_dose) %>% 
    get_cum_drug %>% 
    get_efficacy %>%
    get_severity_next %>% 
    get_toxicity_group  %>%
    get_outcome
  
  row
}


sim_patient <- function(patient_id, drug_dose){
  MAX_STAY <- 100
  
  history <- list()
  yesterday <- cycle(day_number=0, patient_id=patient_id, drug_dose=drug_dose)
  for (day_number in 1:MAX_STAY){
    today <- cycle(day_number=day_number, patient_id=patient_id, drug_dose=drug_dose, yesterday)
    history <- append(history, list(today))
    if (today[['outcome']] != 'none'){
      break
    }
    yesterday <- today
  }
  
  do.call('rbind', history) %>% as.data.frame
}


# OUTPUT_DIR <- 'simsim_patients'
sim_population <- function(num_patients, next_pid=1){
  patient_list = list()
  for (patient_id in next_pid:(next_pid + num_patients - 1)){
    dosage_level <- patient_id %% NUM_COHORTS
    drug_dose <- sprintf('c%02d', dosage_level)
    patient <- sim_patient(patient_id, drug_dose)
#     outfile <- sprintf("%s/patient_%03d.csv", OUTPUT_DIR, patient_id)
#     print(outfile)
#     write.csv(patient, outfile)
    patient_list = append(patient_list, list(patient))
  }
  bind_rows(patient_list)
}


# COMMAND ----------

simsimpop <- sim_population(num_patients=NUM_COHORTS * SAMPLES_PER_COHORT) # 800 took 8.89 minutes


# COMMAND ----------

plot_dose_survival <- function(sspop){
  
  last_day <- sspop %>% 
    mutate(cohort = as.numeric(gsub('c','', drug))) %>% 
    group_by(patient_id) %>% 
    arrange(as.numeric(day_number)) %>%
    filter(day_number == last(day_number)) %>% 
    ungroup
  
  last_day %>% 
    group_by(cohort) %>% 
    summarize(survival=sum(outcome=='recover')/n(), n=n()) %>%
    mutate(dose=cohort/10) %>% 
    ggplot(aes(x=dose, y=survival)) + geom_line()
  
}

# COMMAND ----------

# require(SparkR)

# simsimpop %>% 
#   SparkR::createDataFrame() %>%
#   SparkR::saveAsTable(tableName="bogovirus.simsim_population_cat10", source="parquet", mode="overwrite") 

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC 
# MAGIC simsim_population = spark.sql("select * from bogovirus.simsim_population_cat10").toPandas()
# MAGIC 
# MAGIC # What was the assigned dose for each patient?
# MAGIC patient_dose = simsim_population[ ['patient_id', 'drug'] ].groupby('patient_id').max()
# MAGIC 
# MAGIC simsim_population['cohort'] = [patient_dose['drug'][pid] for pid in simsim_population.patient_id.values]
# MAGIC simsim_population['cohort_dose'] = [ float(patient_dose['drug'][pid].replace('c', ''))/10 for pid in simsim_population.patient_id.values ]
# MAGIC 
# MAGIC # What fraction of each cohort died?
# MAGIC last_day = simsim_population.groupby('patient_id').last()
# MAGIC last_day['survive'] = [1 if x == 'recover' else 0 for x in last_day['outcome']]
# MAGIC 
# MAGIC # last_day
# MAGIC 
# MAGIC cohort_survive = last_day[ ['cohort', 'survive'] ].groupby('cohort').mean().reset_index()
# MAGIC cohort_dose = simsim_population[ ['cohort', 'cohort_dose'] ].groupby('cohort').max().reset_index()
# MAGIC 
# MAGIC 
# MAGIC dose_outcome_pdf = pd.concat([cohort_dose, cohort_survive], axis=1)
# MAGIC 
# MAGIC print("Best survival fraction:", np.max(dose_outcome_pdf.survive))
# MAGIC 
# MAGIC import matplotlib.pyplot as plt
# MAGIC 
# MAGIC ax = dose_outcome_pdf.plot(x='cohort_dose', y='survive', 
# MAGIC                       # xlabel='Dose of drug', ylabel='Fraction of cohort surviving', 
# MAGIC                       title="Optimizing dose with extrapolated 'simsim'", legend=False, 
# MAGIC                       figsize=(12, 8), fontsize=18, lw=6)
# MAGIC ax.title.set_size(36)
# MAGIC ax.set_xlabel('Dose of drug',fontdict={'fontsize':24})
# MAGIC ax.set_ylabel('Fraction of cohort surviving',fontdict={'fontsize':24})
# MAGIC 
# MAGIC best_dose = dose_outcome_pdf.cohort_dose[np.argmax(dose_outcome_pdf.survive)]
# MAGIC plt.axvline(x=best_dose, linestyle='dashed', color='red')

# COMMAND ----------

# MAGIC %python
# MAGIC dose_outcome_pdf

# COMMAND ----------


