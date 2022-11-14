# reduce trajectory py
# JMA 5 Oct 2022
# Scan all trajectory records to compute survivals
import os, re, sys
from pathlib import Path
import pandas as pd

def parse_filename(a_fn: Path):
    base = a_fn.name
    m = re.search(r'_(\d+)_(\S+)\.csv$', str(base))
    nn = int(m.group(1))
    dose = float(m.group(2))
    return nn, dose


# Extract terminal episode records from a file.
def get_terminal_records(a_df):
    ''
    # Select all terminal records
    terminals = a_df.loc[abs(a_df.Reward) > 1,:]
    # Compute return value by terminal

    # Compute a summary survival
    return terminals.iloc[:,1] - terminals.iloc[:,0]

if __name__ == '__main__':

    home = Path(sys.argv[1])
    the_d = []
    for a_file in home.glob('traj*.csv'):
        nn, dose = parse_filename(a_file)
        a_df = pd.read_csv(a_file, header=0)
        the_d.append([get_terminal_records(a_df).mean(), nn, dose])
        print(a_file)
    pd.DataFrame(the_d, columns=["Q", 'nn', 'dose']).to_csv('All_trajectories.csv')
