import gymnasium as gym
import highway_env
from utils import record_videos, show_videos
import pprint
import matplotlib.pyplot as plt
import numpy as np
import os
import agent
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#************************************************************************
env = gym.make('racetrack-v0', render_mode='rgb_array')
# env.close()
env.configure({
    'action': {'lateral': True,
            'longitudinal': False,
            'target_speeds': [0, 5],
            'type': 'ContinuousAction'},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 2,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", 
                     "heading", "long_off", "lat_off", "ang_off"],
    },
    "other_vehicles": 1,
    'show_trajectories': True,
     'offroad_terminal': True,
})
# pprint.pprint(env.config)
(obs, info), done = env.reset(), False
obs = np.array(obs.flatten())
print("Environment is setted up.")
#************************************************************************


#************************************************************************
agent = agent.DDQNAgent(alpha = 0.001, gamma = 0.999, epsilon = 1.0, 
                       obs_shape = obs.shape, batch_size = 64, epsilon_dec = 0.993, 
                       epsilon_end = 0.05, mem_size = 100000, min_mem_size = 100, 
                       replace_target = 1000, learning_rate=0.001)
print("Agent is initialized.")
#************************************************************************

best_score = -1000.0
score_history = []

for episode in range(2000):
        (observation, info), done = env.reset(), False
        observation = np.array(observation.flatten())

        if (episode % 500 == 0) and (episode != 0):
            env = record_videos(env)

        done_ = False
        score = 0
        step = 0
        # env.render()
        while True:
            action, action_index = agent.get_action(observation.reshape((1,observation.shape[0])))
            new_observation, reward, done, truncated, info = env.step(action=[action])
            new_observation = np.array(new_observation.flatten())

            if info["crashed"] == True or info["rewards"]["on_road_reward"] == False:
                done_ = True
                reward = -1.0
            else: done_ = False

            score +=reward
            agent.remember(state=observation, action=action_index, done=done,
                           reward=reward, new_state=new_observation)
            agent.train()

            observation = new_observation

            if done or done_:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        print('episode ', episode, 'score %.1f' % score,
               'avg score %.1f' % avg_score)

env.close()
# show_videos()


