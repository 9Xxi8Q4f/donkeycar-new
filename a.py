import gymnasium as gym
import highway_env
from utils import record_videos, show_videos
import pprint
import matplotlib.pyplot as plt

env = gym.make('racetrack-v0', render_mode='rgb_array')
# env.config["real_time_rendering"] = True
pprint.pprint(env.config)
(obs, info), done = env.reset()

env = record_videos(env)
for episode in range(3):
    (obs, info), done = env.reset(), False
    print(obs)
    print(info)
    env.render()
    while not done:
        action = 1.0
        obs, reward, done, truncated, info = env.step([action])
env.close()
show_videos()

