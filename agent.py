from per import ReplayBuffer as buffer 
import numpy as np
from keras.utils import plot_model
# from network import QNetwork as q_network
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
tf.random.set_seed(43)

class DDQNAgent:

    def __init__(self, alpha = None, gamma = None, epsilon = None, obs_shape = None,
                 batch_size = None, epsilon_dec = None, epsilon_end = None, mem_size = None, 
                 min_mem_size = None, learning_rate = None, replace_target = None):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.min_mem_size = min_mem_size
        self.replace_target = replace_target
        self.obs_shape = obs_shape
        self.learning_rate = learning_rate

        self.discrete_action_space = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        self.n_actions = len(self.discrete_action_space)
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = buffer(max_size=self.mem_size, min_size=self.min_mem_size,input_shape=self.obs_shape,
                             n_actions=self.n_actions,discrete=True)
                        
        self.q_eval = self._make_model()
        self.q_target = self._make_model()      #we keep a target model which we update every K timesteps
        self.q_eval.summary()
        plot_model(self.q_eval, to_file='model_ddqn.png')

    def _make_model(self):
        
        model = Sequential()
        model.add( Dense(64, input_dim = self.obs_shape[0], activation='relu') )
        # model.add( Dense(512, activation='relu') )
        model.add( Dense( self.n_actions))
        model.compile(loss='mse',optimizer= Adam(learning_rate = self.learning_rate),metrics=["accuracy"])
 
        return model

    def epsilon_decay(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end \
        else self.epsilon_end

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())
        
    def get_action(self, observation):

        if np.random.random() > self.epsilon:
            qs_= self.q_eval.predict(observation)
            action_index = np.argmax(qs_)
            action = self.discrete_action_space[action_index]
        else:
            action_index = np.random.randint(0, self.n_actions)
            action = self.discrete_action_space[action_index]
        
        return action, action_index

    def train(self):

        if (self.memory.mem_cntr) < self.min_mem_size:
            return
        #* and ELSE:
        #* sample minibatch and get states vs..
        state, action, reward, new_state, done, sample_indices = \
                            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        #* get the q values of current states by main network
        q_pred = self.q_eval.predict(state)

        #! for abs error
        target_old = np.array(q_pred)

        #* get the q values of next states by target network
        q_next = self.q_target.predict(new_state) #! target_val

        #* get the q values of next states by main network
        q_eval = self.q_eval.predict(new_state) #! target_next

        #* get the actions with highest q values
        max_actions = np.argmax(q_eval, axis=1)

        #* we will update this dont worry
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #* new_q = reward + DISCOUNT * max_future_q
        q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        #* error
        error = target_old[batch_index, action_indices]-q_target[batch_index, action_indices]
        self.memory.set_priorities(sample_indices, error)

        #* now we fit the main model (q_eval)
        _ = self.q_eval.fit(state, q_target, verbose=0)

        #* If counter reaches set value, update target network with weights of main network
        #* it will update it at the very beginning also
        if self.memory.mem_cntr & self.replace_target == 0:
            self.update_network_parameters()
            print("Target Updated")

        self.epsilon_decay()

    def save_model(self):
        print("-----saving models------")
        self.q_eval.save_weights("saved_models/q_net.h5")
        # self.q_target.save_weights(self.network.checkpoint_file)


