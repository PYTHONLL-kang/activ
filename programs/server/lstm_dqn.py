# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
import math

from collections import deque
from sklearn.preprocessing import MinMaxScaler

# %%
import os
os.chdir('../../')

# %%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

np.set_printoptions(precision=6, suppress=True)

# %%
real_data = pd.read_excel('./documents/nov_nine_var.xlsx').to_numpy()
goal_data = pd.read_excel('./documents/result/basic_formula.xlsx').to_numpy()

scaler = MinMaxScaler()
scaler = scaler.fit(real_data[:,1:22])

# %%
def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])

def argmin(l):
    return min(range(len(l)), key=lambda i: l[i])

# %%
start = scaler.transform(real_data[:,1:22])[-1].reshape(1, 21)
goal = scaler.transform(goal_data[:,1:22])[argmin(goal_data[:,-1])].reshape(1, 21)

print(goal[0])
print(start[0])

# %%
lstm_state = scaler.transform(real_data[:,1:22])[-13:-1].reshape(1, 12, 21)

# %%
# dqn paramater
GAMMA = 0.99
EPS_DECAY = 0.0005
BATCH_SIZE = 64
EPISODE_DONE = 1000
TRAIN_FLAG = EPISODE_DONE * 10
MEMORY_SIZE = TRAIN_FLAG * 10

LEARN_FREQ = 50
ACTION_NUM = 5

# %%
model_list = [
    [
        tf.keras.models.load_model('./model/one_lstm/one_lstm_{0}/{1}_model'.format(j, i)) for i in range(21)
    ]   for j in range(ACTION_NUM)
]

# %%
def shift_data(origin, d):
    shift_d = np.zeros((1, 12, 21))
    for i in range(21):
        d_s = d[:,i]
        shift_d[0][:,i] = np.concatenate((origin[0][1::][:,i], d_s), axis=0).reshape(1, 12)
    return shift_d

# %%
def return_action(idx, s):
    model_pred = np.zeros((5, 21, 1))
    for i in range(5):
        for j in range(21):
            s_s = s[:,:,j].reshape(1, 12, 1)
            model_pred[i][j] = model_list[i][j](s_s)[0]
    return model_pred[idx].T

# %%
def return_state(s, a):
    ns = s[:,0:21] + a
    return ns

# %%
def return_reward(ns, gs):
    ns_s = ns[:,0:21]
    dist = np.sqrt(np.sum(np.square(gs - ns_s)))

    end = 0
    for i in range(21):
        if ns_s[0][i] == gs[0][i]:
            end += 4
    
    reward = -dist + end
    return reward

# %%
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.8
    beta = 0.3
    beta_increment_per_sampling = 0.0005

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        a_is = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = rand.uniform(a, b)
            (a_i, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            a_is.append(a_i)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, a_is, is_weight

    def update(self, a_i, error):
        p = self._get_priority(error)
        self.tree.update(a_i, p)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, a_i, change):
        parent = (a_i - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, a_i, s):
        left = 2 * a_i + 1
        right = left + 1

        if left >= len(self.tree):
            return a_i

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        a_i = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(a_i, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, a_i, p):
        change = p - self.tree[a_i]

        self.tree[a_i] = p
        self._propagate(a_i, change)

    # get priority and sample
    def get(self, s):
        a_i = self._retrieve(0, s)
        dataa_i = a_i - self.capacity + 1

        return (a_i, self.tree[a_i], self.data[dataa_i])

# %%
class DQN_Network(tf.keras.models.Model):
    def __init__(self):
        super(DQN_Network, self).__init__()
        self.input_layer = tf.keras.layers.Dense(128, activation='relu')

        self.q_layer = tf.keras.models.Sequential()
        self.q_layer.add(tf.keras.layers.Dense(128, activation='relu'))
        self.q_layer.add(tf.keras.layers.Dense(128, activation='relu'))
        self.q_layer.add(tf.keras.layers.Dense(1, activation='linear'))

        self.adv_layer = tf.keras.models.Sequential()
        self.adv_layer.add(tf.keras.layers.Dense(128, activation='relu'))
        self.adv_layer.add(tf.keras.layers.Dense(128, activation='relu'))
        self.adv_layer.add(tf.keras.layers.Dense(ACTION_NUM, activation='linear'))
    
    def call(self, x):
        i = self.input_layer(x)

        q = self.q_layer(i)
        adv = self.adv_layer(i)

        o = q + adv - tf.math.reduce_mean(adv, axis=1, keepdims=True)
        return o

# %%
class DQN_Agent:
    def __init__(self):
        self.train_model = self.set_model()
        self.target_model = self.set_model()
        self.target_model.trainable = False

        self.memory = Memory(MEMORY_SIZE)
        self.episode = 1
        self.eps_threshold = 1

        self.optim = tf.keras.optimizers.RMSprop(learning_rate=1e-11)

    def set_model(self):
        net = DQN_Network()
        net.build(input_shape=(1, 42))

        optim = tf.keras.optimizers.RMSprop(learning_rate=1e-11)
        net.compile(optimizer=optim, loss='mse')
        return net

    def update_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    def soft_update_model(self):
        train_weight = np.array(self.train_model.get_weights(), dtype=object)
        target_weight = np.array(self.target_model.get_weights(), dtype=object)

        weight = train_weight * 0.01 + target_weight * 0.99
        self.target_model.set_weights(weight)

    def memorize(self, cs, a_i, r, ns, d):
        if d and self.memory.tree.n_entries > TRAIN_FLAG:
            self.episode += 1

        td_error = r + GAMMA * np.argmax(self.target_model(ns)[0]) - np.argmax(self.train_model(cs)[0])
        self.memory.add(td_error, (cs, a_i, r, ns, d))

    def convert_memory_to_input(self, batch):
        s, a_i, r, ns, d = zip(*batch)

        states = tf.convert_to_tensor(s).reshape(BATCH_SIZE, 42)
        action_indexs = tf.convert_to_tensor(a_i)
        rewards = tf.convert_to_tensor(r)
        next_states = tf.convert_to_tensor(ns).reshape(BATCH_SIZE, 42)
        dones = tf.convert_to_tensor(d)

        return states, action_indexs, rewards, next_states, dones

    def act(self, state):
        a_r = np.array(self.train_model(state))[0]

        if rand.random() > self.eps_threshold:
            a_i = np.argmax(a_r)
            c = 1

        else:
            a_i = rand.randint(0, ACTION_NUM-1)
            c = 0

        return a_i, c, self.eps_threshold

    def run(self):
        if self.memory.tree.n_entries < TRAIN_FLAG:
            return 1
        
        self.eps_threshold = 0.05 + (1 - 0.05) * math.exp(-1. * self.episode * EPS_DECAY)

        batch, a_is, is_weight = self.memory.sample(BATCH_SIZE)

        states, action_indexs, rewards, next_states, dones = self.convert_memory_to_input(batch)
        is_weight = tf.convert_to_tensor(is_weight)
        loss = self.learn(states, action_indexs, rewards, next_states, dones, is_weight)

        return loss.numpy()

    @tf.function
    def learn(self, states, action_indexs, rewards, next_states, dones, is_weight):
        with tf.GradientTape() as tape:
            tape.watch(self.train_model.trainable_variables)

            q = self.train_model(states)
            next_q = self.train_model(next_states)
            next_target_q = self.target_model(next_states)

            next_action = tf.argmax(next_q, axis=1)

            target_val = tf.reduce_sum(tf.one_hot(next_action, ACTION_NUM) * next_target_q, axis=1)
            target_q = rewards + (1 - dones) * GAMMA * target_val

            main_val = tf.reduce_sum(tf.one_hot(action_indexs, ACTION_NUM) * q, axis=1)

            error = tf.square(main_val - target_val) * 0.5
            loss = tf.reduce_mean(error)

        grads = tape.gradient(loss, self.train_model.trainable_weights)
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.optim.apply_gradients(zip(grads, self.train_model.trainable_weights))

        return loss

# %%
agent = DQN_Agent()
state_hist = []
reward_hist = [[] for i in range(4)]
loss_hist = []
eps_hist = []
steps_list = []

for e in range(20000 + TRAIN_FLAG // EPISODE_DONE):
    state = np.array([start, goal]).reshape(1, 42)
    steps = 1
    reward = return_reward(state, goal)
    rewards = 0

    while True:
        lstm_state = shift_data(lstm_state, state[:,0:21])
        a_i, t, eps = agent.act(state)
        action = return_action(a_i, lstm_state)

        checker = state[:,0:21] == goal[0]
        if steps == EPISODE_DONE or all(checker[0]):
            done = 1
        else:
            done = 0

        next_state = np.array([return_state(state, action), goal]).reshape(1, 42)
        reward = return_reward(next_state, goal)

        agent.memorize(state, a_i, reward, next_state, done)
        if steps % LEARN_FREQ == 0:
            loss = agent.run()
            agent.soft_update_model()
        
        state = next_state
        rewards += reward
        steps += 1

        if done:
            rewards = rewards if steps - 1 == EPISODE_DONE else -100
            reward_hist[0].append(rewards)
            print(f'============={e if steps -1 == EPISODE_DONE else 0}=============')

            break

# %%



