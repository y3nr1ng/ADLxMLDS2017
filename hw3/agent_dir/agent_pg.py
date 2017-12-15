from agent_dir.agent import Agent

from collections import namedtuple

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from keras.utils import print_summary, to_categorical

from pprint import pprint
np.set_printoptions(threshold=np.inf)

class Agent_PG(Agent):
    WEIGHT_FILE = 'agent_pg_weight.h5'
    History = namedtuple('History',
                         ['state', 'probability', 'gradient', 'reward'])

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

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
        model = Sequential([
            Conv2D(16, (8, 8), activation='relu', strides=(4, 4), input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)),
            Conv2D(32, (4, 4), activation='relu', strides=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.env.get_action_space().n, activation='softmax')
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

    def train(self):
        """
        Implement your training algorithm here
        """
        avg_score = None
        for i_ep in range(10000):
            score, loss = self._train_once()
            if avg_score is None:
                avg_score = score
            else:
                avg_score = avg_score*0.99 + score*0.01
            print('ep {}, score(avg) = {}({:.06f}), loss = {:.06f}'.format(i_ep, score, avg_score, loss))
            if i_ep % 100 == 0:
                self.model.save_weights(Agent_PG.WEIGHT_FILE)
                print('...saved')
        self.model.save_weights(Agent_PG.WEIGHT_FILE)

    def _train_once(self, gamma=0.99, lr=1e-3):
        """
        Train a single episode.
        """
        # initialize
        observation = self.env.reset()
        score = 0
        history = []

        # run a single episode
        done = False
        prev_field = None
        t = 0
        while not done:
            # execute a step
            action, p_action = self.make_action(observation, test=False)
            #pprint('{} {}'.format(['{0:.3f}'.format(i) for i in p_action], action))
            observation, reward, done, _ = self.env.step(action)

            score += reward

            curr_field, player = Agent_PG._preprocess(observation)
            if prev_field is None:
                prev_field = np.zeros_like(curr_field)
            # concat all the variables
            state = np.hstack([player, curr_field, prev_field])
            prev_field = curr_field

            # calculate probability gradient
            p_decision = to_categorical(action,
                                        num_classes=self.env.get_action_space().n)
            gradient = p_decision.astype(np.float32) - p_action

            # remember the result
            entry = Agent_PG.History(state, p_action, gradient, reward)
            history.append(entry)

        # extract the results
        states, probability, gradients, rewards = zip(*history)

        # discount rewards
        rewards = np.vstack(rewards)
        rewards = self._discount_rewards(rewards, gamma=gamma)
        # normalize
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        # attenuate the gradients
        gradients = np.vstack(gradients)
        gradients *= rewards

        # batch training
        X = np.vstack(states)
        probability = np.vstack(probability)
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

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        curr_field, player = Agent_PG._preprocess(observation)
        if self._prev_field is None:
            self._prev_field = np.zeros_like(curr_field)
        # concat all the variables
        state = np.hstack([player, curr_field, self._prev_field])
        self._prev_field = curr_field

        state = np.expand_dims(state, axis=0)
        p_action = self.model.predict(state, batch_size=1).flatten()
        # normalize the PDF
        p_action /= np.sum(p_action)

        # determine the action according to the PDF
        action = np.random.choice(self.env.get_action_space().n, p=p_action)
        if test:
            return action
        else:
            return action, p_action

    @staticmethod
    def _preprocess(observation):
        """
        Process (split fields and segmented) the observation field.

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            observation: np.array
                processed binary image, only play field and our board
        """
        # convert red layer binarized float
        observation = observation[..., 2].astype(np.float32)
        observation[observation < 50] = 0.
        observation[observation >= 50] = 1.

        # field
        #   x=20, y=34, w=120, h=160
        field = observation[34:194:2, 20:140:2]
        player = observation[34:194:2, 140]

        # apply the weight
        weights = np.arange(1, 61, dtype=np.float32) / 60
        field *= weights
        field = np.sum(field, axis=1)

        return field, player
