'''
DeepQNetworkAgentClass.py

Implementation for a Deep Q Network learner

From:
    Mnih, Volodymyr et al. Human-level control through deep
    reinforcement learning. Nature 518 7540 (2015): 529-33.

Based on DQN implementation by Christopher Grimm and Melrose Roderick
'''
# Python imports.
from __future__ import division
import numpy as np
import tensorflow as tf

# Local imports.
from ..AgentClass import Agent

# TODO: Make ReplayMemory, update Gym MDP + State classes, test
class DeepQNetworkAgent(Agent):
    '''
    Agent that uses a deep convolutional neural network as a nonlinear Q-function approximator.
    '''

    def __init__(self, state_shape, state_dtype, num_actions, name="DQN", gamma=0.99, learning_rate=0.00025, batch_size=32,
                 replay_memory_start=50000, replay_memory_max=1000000, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_steps=1000000, update_freq=4, target_copy_freq=10000, history_size=4, error_clip=1):
        '''
        Args:
            state_shape (list): Expected shape of observations (Numpy arrays)
            state_dtype (str): The data type associated with values stored in observation arrays
            num_actions (int): The total number of actions available to the agent
            gamma (float): Discount factor
            learning_rate (float): Learning rate
            replay_memory_start (int): Steps to run in the environment under a purely random policy before learning
            replay_memory_max (int): The maximum number of transitions to store in replay memory
            epsilon_start (float): Initial chance of randomness
            epsilon_end (float): Final annealed value of epsilon
            epsilon_steps (int): The number of steps desired for annealing epsilon from epsilon_start to epsilon_end
            update_freq (int): The number of actions to execute before performing a training step on the network
            target_copy_freq (int): The number of steps to take before updating target network weights to those of the current network
            history_size (int): The total number of frames that should be stacked together to compose a single state
            batch_size (int): The total number of samples in a minibatch of training
            error_clip (float): Clipping term for the TD(0) error when training the network
        '''
        Agent.__init__(self, name=name, actions=range(num_actions), gamma=gamma)
        # Tensorflow configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # DQN configuration
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.num_actions = num_actions
        self.action_counter = 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_memory_start = replay_memory_start
        self.replay_memory_max = replay_memory_max
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.epsilon_delta = (self.epsilon - self.epsilon_end) / self.epsilon_steps
        self.update_freq = update_freq
        self.target_copy_freq = target_copy_freq
        self.history_size = history_size
        self.error_clip = error_clip
        self.replay_buffer = ReplayMemory(self.state_shape, self.state_dtype, self.replay_memory_max, self.history_size)

        # Setup input placeholders
        self.init_placeholders()
        # Setup forward pass pipeline for DQN and target DQN
        self.init_networks()
        # Setup loss computation
        self.compute_loss()
        # Setup training operation
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, centered=True, epsilon=0.01)
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.get_variables_with_scope('main'))
        # Setup operation for target network updates
        self.target_update_op = self.swap_variables(source_scope='main', dest_scope='target')

        # Initialize the Tensorflow computation graph and variables
        self.sess.run(tf.global_variables_initializer())

    def run_update_step(self):
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(self.batch_size)
        one_hot_actions = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        one_hot_actions[range(len(actions)), actions] = 1

        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.state: states, self.action: one_hot_actions,
                                           self.next_state: next_states, self.reward: rewards,
                                           self.terminal: terminals})

        return loss

    def get_max_q_action(self, state):
        q_values = self.sess.run(self.main_network, feed_dict={self.state: [state]})
        return np.argmax(q_values[0])

    def epsilon_greedy_q_policy(self, state):
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if np.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = np.random.choice(range(self.num_actions))

        return action

    def act(self, state, reward):
        # Select action according to an epsilon-greedy Q-policy
        action = self.epsilon_greedy_q_policy(state)

        # If we have a complete s, a, r, s', t tuple
        if self.prev_state and self.prev_action:
            # Perform epsilon annealing
            if self.replay_buffer.size() > self.replay_memory_start:
                self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_delta)

            next_state, is_terminal = state, state.is_terminal()
            state, action = self.prev_state, self.prev_action

            # Add most recent experience to replay memory
            self.replay_buffer.append(state[-1], action, reward, is_terminal)

            # Perform a single update step on the network
            if (self.replay_buffer.size() > self.replay_memory_start) and (self.action_counter % self.update_freq == 0):
                loss = self.run_update_step()

            # Copy main network weights over to the target network when appropriate
            if (self.action_counter - self.replay_memory_start) % self.target_copy_freq == 0:
                self.sess.run(self.target_update_op)

        self.prev_state = state
        self.prev_action = action
        self.action_counter += 1
        return action


    # ---------------------------------
    # ---- TENSORFLOW ----
    # ---------------------------------

    def get_variables_with_scope(self, *scopes):
        return map(lambda x: tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=x+'/'), scopes)

    def swap_variables(self, source_scope, dest_scope):
        return [tf.assign(dv, sv) for sv, dv in zip(self.get_variables_with_scope(source_scope), self.get_variables_with_scope(dest_scope))]

    def conv2d(self, input, kernel, stride, num_in_filters, num_out_filters, rectifier=lambda x:x):
        with tf.variable_scope('conv_layer_params'):
            W = tf.get_variable('W', [kernel, kernel, num_in_filters, num_out_filters],
                                initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('B', [num_out_filters], initializer=tf.constant_initializer(0.0))
        conv_out = rectifier(tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME') + B)
        return conv_out

    def fully_connected(self, input, num_outputs, rectifier=lambda x:x):
        with tf.variable_scope('fc_layer_params'):
            W = tf.get_variable('W', [input.get_shape()[1].value, num_outputs],
                                initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('B', [num_outputs], initializer=tf.constant_initializer(0.0))
        fc = rectifier(tf.matmul(input, W) + B)
        return fc

    def init_placeholders(self):
        self.state = tf.placeholder(self.state_dtype, [None] + list(self.state_shape) + [self.history_size])
        self.action = tf.placeholder(tf.float32, [None, self.num_actions])
        self.next_state = tf.placeholder(self.state_dtype,  [None] + list(self.state_shape) + [self.history_size])
        self.terminal = tf.placeholder(tf.bool, [None])
        self.reward = tf.placeholder(tf.float32, [None])

    def init_networks(self):
        with tf.variable_scope('main'):
            self.main_network = self.get_dqn(self.state)
        with tf.variable_scope('target'):
            self.target_network = self.get_dqn(self.next_state)

    def compute_loss(self):
        # Clip rewards to be between -1 and 1
        self.clipped_reward = tf.minimum(tf.maximum(self.reward, tf.constant(1.0)), tf.constant(-1.0))
        is_terminal = tf.cast(tf.logical_not(self.terminal), dtype=tf.float32)
        self.maxQ = tf.reduce_max(self.target_network, reduction_indices=1)
        self.td_zero_target = self.r + is_terminal * self.gamma * self.maxQ
        self.td_zero_error  = tf.reduce_sum(self.action * self.main_network, reduction_indices=1) - self.td_zero_target
        self.clipped_error = tf.where(tf.abs(self.td_zero_error) < self.error_clip, 0.5 * tf.square(self.td_zero_error),
                                      self.error_clip * tf.abs(self.td_zero_error))

        self.loss = tf.reduce_sum(self.clipped_error)
        self.gradients = tf.gradients(self.loss, self.main_network)

    def get_dqn(self, input_place):
        input = tf.image.convert_image_dtype(input_place, tf.float32)
        with tf.variable_scope('C1'):
            c1 = self.conv2d(input, kernel=8, stride=4, num_in_filters=self.history_size, num_out_filters=32, rectifier=tf.nn.relu)
        with tf.variable_scope('C2'):
            c2 = self.conv2d(c1, kernel=4, stride=2, num_in_filters=32, num_out_filters=64, rectifier=tf.nn.relu)
        with tf.variable_scope('C3'):
            c3 = self.conv2d(c2, kernel=3, stride=1, num_in_filters=64, num_out_filters=64, rectifier=tf.nn.relu)
            c3 = tf.reshape(c3, [-1] + reduce(lambda x,y: x * y, c3.get_shape().as_list()[1:]))
        with tf.variable_scope('FC1'):
            fc1 = self.fully_connected(c3, num_outputs=512, rectifier=tf.nn.relu)
        with tf.variable_scope('O'):
            q_values = self.fully_connected(fc1, num_outputs=self.num_actions)
        return q_values

class ReplayMemory():
    def __init__(self, obs_shape, obs_dtype, capacity, history):
        self.index = 0
        self.full = False
        self.capacity = capacity
        self.history = history
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.numpy_transpose_shape = range(1, len(self.obs_shape) + 1) + [0]

        self.obs = np.zeros([self.capacity] + list(self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terms = np.zeros(self.capacity, dtype=np.bool)

    def append(self, s, a, r, t):
        # Notice that the next state will be stored at (self.index + 1) % self.capacity
        self.obs[self.index, :] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.terms[self.index] = t

        if self.index + 1 >= self.capacity:
            self.full = True
        self.index = (self.index + 1) % self.capacity

    def get_index_sample(self, ind):
        state_terms = self.terms[ind-self.history:ind]
        # Until we can get history number of consecutive frames with no interrupting terminal states
        while state_terms.any() or not state_terms[-1]:
            ind -= 1
            state_terms = self.terms[ind - self.history:ind]

        state = np.transpose(self.obs[ind-self.history:ind], self.numpy_transpose_shape)
        next_state = np.transpose(self.obs[ind-self.history+1:ind+1], self.numpy_transpose_shape)
        return state, self.actions[ind], self.rewards[ind], next_state, self.terms[ind]

    def sample(self, num_samples):
        if not self.full:
            # Only attempt to sample the data we have
            idx = np.random.randint(self.history-1, self.index, size=num_samples)
        else:
            # Otherwise, avoid hitting the end of the buffer
            idx = np.random.randint(self.history - 1, self.capacity, size=num_samples)
            # idx = idx - self.index - self.history - 1
            # idx = idx % self.capacity
        idx = list(idx)

        batch = [self.get_index_sample(x) for x in idx]
        batch_states = np.array([_[0] for _ in batch])
        batch_actions = np.array([_[1] for _ in batch])
        batch_rewards = np.array([_[2] for _ in batch])
        batch_next_states = np.array([_[3] for _ in batch])
        batch_terminals = np.array([_[4] for _ in batch])

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals

    def size(self):
        return self.capacity if self.full else self.index
