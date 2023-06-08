# README bogovirus/beta

Simulation of the Beta strain of _Bogovirus_.

Bogovirus_beta_sim.ipynb
Bogovirus_β.pptx
README.txt                            # This file

benvs/
    benvs/offline:
        BogoOfflineEnv.py
        __init__.py
        timevarying_policy.py

    benvs/online:
        BogoBetaEnv.py               # beta_simulation - the object for online env
        __init__.py

    benvs/discrete                   # An online version with discretized observable states 
        BogoDiscreteEnv.py
        __init__.py

    benvs/policies:
        BogoPolicies.py              # Object with different policies used by 
        __init__.py                  # other environments. 

beta_simulation.py                   # Test the procedural simulation (since converted into the online env)
cnt_100_patients.csv
cohort_const.csv
generate_primary_sim.py              # Run the online RL experiments.
iter_run_timevarying.py
model_UDF_example.ipynb
readme.md
run_beta.ipynb
run_beta_batch.ipynb
run_beta_offline.ipynb
run_beta_online.ipynb