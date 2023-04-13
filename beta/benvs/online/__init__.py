# module for a gym environment
# that wraps an off-line data set. 
from gym.envs.registration import register

register(
    id='BogoEnvBeta-Acc-v0',
    entry_point='envs_beta.bogo_beta:BogoEnv',
    max_episode_steps=300
)