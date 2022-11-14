README: RL offline  and online simulation

files:

iter_runenv.py                 - Run the offline "Bogovirus" environment and iterate over a range of drug doses
iter_run_accurate.py           - Run the on-line simulation of the environment, over the range of constant drug doses
plot_dose_response.ipynb       - Plot dose-response from trajectory files, for constant dose policies
reduce_trajectory.py           - Extract survival data from a set of trajectory files
run_env_accurate.py            - Run the simulation on-line environment with constant policies
runenv.py                      - Run one episode, to test offline env
simulation.py                  - The online simulation 
transition.py                  - Compute a discrete  probability transition matrix (this wasn't used).

envs/                          - OpenGym module for the offline simulation environment
    __init__.py                - module file
    bogo_world.py              - BogoEnv Class

envs_accurate/                 - OpenGym module wrapping the simulation code
     __init__.py
     bogo_accurate.py

README.txt                     - This file
