#!/usr/bin/env python
# coding: utf-8

# ## Курсовой
#
# https://www.gymlibrary.dev/environments/classic_control/mountain_car/

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
# Импортируем все регуляризаторы кераса
from tensorflow.keras.regularizers import *
import matplotlib.pyplot as plt
import time
# Модуль для сохранения результатов в файл
#import pickle


# Используем больше потоков для обучения нейросети
##session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
##sess = tf.compat.v1.Session(config=session_conf)


# Инициализация среды (поля)
##env = gym.make('MountainCar-v0')
##
### Проверка
##print(env.reset())
##print(env.action_space)
##print(env.action_space.n)
##print(env.observation_space.shape)
##print(env.step(1))
##

# ### Нейросеть 32-16-3
# Создадим нейросеть с 1 входным слоем (по числу сенсоров), 3 скрытыми слоями по 512, 256 и 128 нейрона и 1 выходным - по числу возможных действий

def Make_DQN(input_shape, action_size, _learning_rate):
    X_input = Input(input_shape)
    #     X = Dense(512, input_shape=(input_shape), activation="relu", kernel_initializer='he_uniform', kernel_regularizer = l2(1e-4))(X_input)
    #     X = Dense(256, activation="relu", kernel_initializer='he_uniform', kernel_regularizer = l2(1e-4))(X)
    X = Dense(32, input_shape=(input_shape), activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)
    # линейная активация выходного слоя
    X = Dense(action_size, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs = X_input, outputs = X)
    # небольшая скорость обучения для лучшей сходимости

    optimizer=RMSprop(learning_rate=_learning_rate, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=optimizer , metrics=["accuracy"])

    return model

# ### Тестирование стратегии

def plot_game(progress,avg_progress):
    plt.plot(progress)
    plt.plot(avg_progress)
    plt.plot(rewards)
    #plt.show()
    try:
        plt.savefig("plotFig.png")
    except OSError:
        pass


# https://pylessons.com/CartPole-reinforcement-learning/
# Создадим агента с "памятью". Отберем только успешные шаги для обучения

class DQNAgent:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        # by default, Truncation: The length of the episode is 200
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=256) # очередь из "хороших" шагов

        self.EPISODES = 1000
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.996
        self.batch_size = 64 # размер выборки "хороших" шагов
        self.train_start = 64

        self.rew_modifier = 30 # Бонус множитель

        # create main model
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        #         print(f'{state=}, {action=}, {reward=}, {next_state=}, {done=}')
        self.memory.append((state, action, reward, next_state, done))
        #         print(f'{len(self.memory)=}, {self.train_start=} {self.epsilon=}' )
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            prd = self.model.predict(state, verbose=0)
            return np.argmax(prd)

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        # print(len(minibatch))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def run(self, episode):
        self.EPISODES = min(episode, self.EPISODES)  # Количество эпизодов для обучения

        for e in range(self.EPISODES):
            start = time.time()
            state = self.env.reset()  #(seed=22)
            state = np.reshape(state, [1, self.state_size])

            done = False
            rewards_cum = 0
            i = 0
            while not done: # self.epsilon > self.epsilon_min : #было:
                #  self.env.render()
                action = self.act(state)                 #        print(action)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = self.rew_modifier * abs(next_state[0][1])   # этот reward всегда == -1, поэтому не считаем
                elif next_state[0][0] >= 0.5:   # если задача решена
                    reward = 100
                else:
                    reward = 0 # Если закончили по макс. длине эпизода

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                rewards_cum += reward  # суммарная награда за эпизод

                # обучаемся по 1 раз на случайных выборках
                for j in range(1):
                    self.replay()

            progress.append(i)
            avg_progress.append(np.mean(progress))
            rewards.append(rewards_cum)

            print("episode: {}/{}, score: {}, e: {:.2}, reward: {}".format(e, self.EPISODES, i, self.epsilon, rewards_cum))
            print(f'    {np.mean(progress)=}, {time.time()-start}')

#             self.model.save_weights('MountainCar-v0.h5')

            # Записываем статистику в файл через библиотеку pickle:
            #             with open('DQN_stats.txt', 'wb') as f:
            #                 pickle.dump([progress, avg_progress], f)
#             np.save('progress.npy', progress)    # .npy extension is added if not given
#             np.save('avg_progress.npy', avg_progress)    # .npy extension is added if not given
#             print("Статистика успешно сохранена.")

            # каждый 10 эпизод записываем график в файл
            if e%10 == 0 :
                plot_game(progress,avg_progress)


# Запуск обучения


if __name__ == "__main__":
    #env_name = 'PongDeterministic-v4'

    # Создание модели для обучения
    # входные и выходные параметры взяты из документации MountainCar
    model = Make_DQN((2,), 3, 0.005) ## небольшая скорость обучения для лучшей сходимости
    model.summary()

    #load_pretrained = False, если мы обучаем модель с нуля или (True) - продолжаем предыдущую сессию обучения
    load_pretrained = False

    progress = []
    avg_progress = []
    rewards = []

    if load_pretrained:
        model.load_weights('MountainCar-v0.h5')
    #     with open('/content/vizdoom_DQN_stats.txt', 'rb') as f:
    #         record_rewards, record_kills, record_ammos, episode_number, timestep, epsilon = pickle.load(f)
        progress = np.load('progress.npy')    # .npy extension is added if not given
        progress = list(progress)
        avg_progress = list(np.load('avg_progress.npy'))    # .npy extension is added if not given

    agent = DQNAgent()
    agent.run(200)

    # Итоговый график
    plot_game(progress,avg_progress)
