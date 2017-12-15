from agent_dir.agent import Agent

from collections import namedtuple

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from keras.utils import print_summary, to_categorical

#from keras import backend as K
#K.set_image_dim_ordering('th')

from pprint import pprint
np.set_printoptions(threshold=np.inf)

class Agent_PG(Agent):
    WEIGHT_FILE = 'agent_pg_weight.h5'
    History = namedtuple('History',
                         ['state', 'probability', 'gradient', 'reward'])

    FRAME_WIDTH = 80
    FRAME_HEIGHT = 80

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        self._num_actions = self.env.get_action_space().n

        self._build_network()
        if args.test_pg or args.reuse:
            print('load weights from \'{}\''.format(Agent_PG.WEIGHT_FILE))
            self.model.load_weights(Agent_PG.WEIGHT_FILE)
        self._compile_network()

        self._prev_field = None

    def _build_network(self):
        """
        Create a base network.
        """
        size = (Agent_PG.FRAME_WIDTH, Agent_PG.FRAME_HEIGHT, 1)
        model = Sequential([
            Conv2D(16, (8, 8), activation='relu', strides=(4, 4),
                   input_shape=size, data_format='channels_last'),
            Conv2D(32, (4, 4), activation='relu', strides=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self._num_actions, activation='softmax')
        ])
        print_summary(model)
        self.model = model

    def _compile_network(self, lr=1e-3):
        self.model.compile(optimizer=RMSprop(lr=lr, decay=0.99),
                           loss='categorical_crossentropy')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass

    def train(self, num_ep=1, save_interval=500):
        """
        Implement your training algorithm here
        """
        avg_score = None
        for i_ep in range(num_ep):
            score, loss = self._train_once()
            if avg_score is None:
                avg_score = score
            else:
                avg_score = avg_score*0.99 + score*0.01
            print('ep {}, score(avg) = {}({:.06f}), loss = {:.06f}'.format(i_ep, score, avg_score, loss))
            if i_ep % save_interval == 0:
                self.model.save_weights(Agent_PG.WEIGHT_FILE)
                print('...saved')
        self.model.save_weights(Agent_PG.WEIGHT_FILE)

    def _train_once(self, gamma=0.99, lr=1e-3):
        """
        Train a single episode.
        """
        score = 0
        history = []
        prev_state = None

        done = False
        observation = self.env.reset()
        while not done:
            curr_state = Agent_PG._preprocess(observation)
            if prev_state is not None:
                state = curr_state - prev_state
            else:
                state = np.zeros_like(curr_state)
            prev_state = curr_state

            action, p_action = self.make_action(state, test=False)
            observation, reward, done, _ = self.env.step(action)
            score += reward

            # decision gradient
            p_decision = to_categorical(action, num_classes=self._num_actions)
            gradient = p_decision.astype(np.float32) - p_action

            entry = Agent_PG.History(state, p_action, gradient, reward)
            history.append(entry)

        states, probability, gradients, rewards = zip(*history)
        probability = np.vstack(probability)
        gradients = np.vstack(gradients)
        rewards = np.vstack(rewards)

        # discount rewards
        rewards = self._discount_rewards(rewards, gamma=gamma)
        # normalize
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        # attenuate the gradients
        gradients = np.cumsum(gradients, axis=0)
        gradients *= rewards

        # batch training
        X = np.vstack([states])
        Y = probability + lr * gradients
        loss = self.model.train_on_batch(X, Y)

        return score, loss

    def _discount_rewards(self, rewards, gamma=0.99):
        d_rewards = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for i in reversed(range(rewards.size)):
            if rewards[i] != 0:
                running_add = 0
            running_add = running_add * gamma + rewards[i]
            d_rewards[i] = running_add
        return d_rewards

    def make_action(self, state, test=True):
        """
        Return predicted action of your agent
        """
        state = np.expand_dims(state, axis=0)
        p_action = self.model.predict(state, batch_size=1).flatten()

        # determine the action according to the PDF
        action = np.random.choice(self._num_actions, p=p_action)
        if test:
            return action
        else:
            return action, p_action

    @staticmethod
    def _preprocess(observation, size=(80, 80)):
        """
        Preprocess the observation.
        1) Convert to grayscale
        2) Resize
        """
        observation = np.dot(observation[..., :3], [0.2126, 0.7152, 0.0722])
        observation = observation.astype(np.uint8)

        observation = cv2.resize(observation, size)

        return np.expand_dims(observation.astype(np.float32), axis=2)
