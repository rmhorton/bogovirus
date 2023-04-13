# timevarying_policy.py
# Compute a policy as a function of time, solely, with a single mode. 
# Optimize the reward by varying the policy shape parameters. 
# 5 Dec 2022  JMA

import os, re, sys
import math
import pandas as pd
import numpy as np
from numpy.random import default_rng
import datetime as dt
import webbrowser
import shutil

from bokeh.plotting import figure, show, save
from bokeh.models import ColumnDataSource, Text
from bokeh.io import output_file

class TimeVaryingPolicy (object):
    
    my_rng = default_rng(seed=None)
    
    def __init__(self, M: float, last_stage: int) -> None:
        ''
        self.M = M
        self.stage_index = np.arange(last_stage)
        pass
    
    def policy_trajectory(self, a:float, b:float, c:float) -> pd.Series:
        'Compute the policy as a function of the stage index'
        # x - convex up quadratic
        # d - arctan compresses it to 0-1
        x = [c + x*(b + a * x) for x in self.stage_index]
        df = pd.DataFrame({'x': x, 'd': self.M*(np.arctan(x) + math.pi/2)/(math.pi/2)}, index= self.stage_index)

        return df    
    def plot_policy(self, the_policy):
        ''
        one_fig = figure(title="tst", x_axis_type='datetime', plot_width = 800,  plot_height = 600)
        rdf = the_policy.reset_index(names='stage')
        cds = ColumnDataSource(rdf)
        # one_fig._line_stack(x='DateTime', y='HNRSold', source=cds, color='grey')
        one_fig.line(x='stage', y='x', source = rdf, color='red')
        one_fig.line(x='stage', y='d', source = rdf, color='red', line_dash = 'dashed')
        # glyph = Text(x='DateTime', y='HNRSold', text="label", text_color="DeepSkyBlue")
        # one_fig.add_glyph(cds, glyph)
        return one_fig
        
    
if __name__ == '__main__':
    
    policy = TimeVaryingPolicy(1.0, 14)
    
    p = policy.policy_trajectory(-0.2, -0.1, 10)

    path_saved_to = save(policy.plot_policy(p))
    # Copy it to a convenient place, if necessary without spaces in the path
    # since the Chrome browser would choke on the filename
    print(path_saved_to)
    shutil.copy(path_saved_to, 'local.html')
    # This fails in WSL. 
    # webbrowser.open('../local.html')
    # print(p)
