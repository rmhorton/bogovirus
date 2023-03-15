# TODO

1.  create a file with a "primary" simulation using the Standard of Care policy.  This will be used as the online ground truth -- as if this is what is available in the real world. 
    - create a module "beta_simulation" from the current "beta" notebook  (DONE)
    - Embedd the simuation in an env (no need to use gym) & Create a wrapper than runs the simulation
    - Some descrptive stats on it? E.g. coverage of the state space

    NOTE: DON'T USE THE OPENAI GYM ENV. CREATE A COMPATIBLE ONE.

Stack overflow - 
[3:00 PM] Robert Horton
https://stackoverflow.com/questions/23664877/pandas-equivalent-of-oracle-lead-lag-function
Pandas equivalent of Oracle Lead/Lag function

    df['Data_lagged'] = df.groupby(['Group'])['Data'].shift(1)


2.  Run "sim sims" that learn from the primary sim. (These resemble randomized control trials)
    - duplicate the Bogovirus BN analytical sim.
    - run RHINO's causal discovery alg
    - Q: is there a simulation output file (in addition to optimality stats) to share that others can learn from? 

3. Run a naive RL dynamic policy -- offline RL to see how it suffers from the limited statespace

4. Run a full online RL to see what a better dynamic policy could be -- to measure the "offline" gap

5. Comparisons. How much does causal modeling improve over "naive" offline approaches. 


