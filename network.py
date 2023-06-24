import os 
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
tf.random.set_seed(43)


class QNetwork():
    def __init__(self, input_dims = 5, n_actions=25,
    name='q_net', chkpt_dir='./'):
        super(QNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddqn.h5')
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=512, activation='relu')
        self.out = Dense(units=self.n_actions)
    
    def call(self, state):

        observation_value = self.fc1(state)
        observation_value = self.fc2(observation_value)
        out = self.out(observation_value)

        return out