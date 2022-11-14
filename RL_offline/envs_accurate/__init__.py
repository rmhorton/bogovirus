# module for a gym environment
# that wraps an off-line data set. 
from gym.envs.registration import register

register(
    id='BogoEnv-Acc-v0',
    entry_point='envs_accurate.bogo_accurate:BogoEnv',
    max_episode_steps=300
)