from agent_dir.agent import Agent

from collections import namedtuple

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
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
        if args.test_pg:
            self.model.load_weights(Agent_PG.WEIGHT_FILE)
        self._compile_network()

        self._prev_field = None

    def _build_network(self):
        """
        Create a base network.
        """
        # parse network size
        #   in: 3 types of input, (field, player)
        #   out: n actions
        in_dim = 160 * 2
        out_dim = self.env.get_action_space().n

        init = TruncatedNormal()
        model = Sequential([
            Dense(128, input_shape=(in_dim, ), activation='relu', kernel_initializer=init),
            Dense(32, activation='relu', kernel_initializer=init),
            Dense(out_dim, activation='softmax')
        ])
        print_summary(model)
        self.model = model

    def _compile_network(self, lr=1e-3):
        optimizer = Adam(lr=lr)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')

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
        for i_ep in range(200):
            score, loss = self._train_once()
            print('ep {}, score = {}, loss = {}'.format(i_ep, score, loss))
        self.model.save_weights(Agent_PG.WEIGHT_FILE)

    def _train_once(self, gamma=0.99, lr=1e-2):
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
        while not done:
            # execute a step
            action, p_action = self.make_action(observation, test=False)
            #pprint(['{0:.4f}'.format(i) for i in p_action])
            observation, reward, done, _ = self.env.step(action)

            score += reward

            opponent, curr_field, player = Agent_PG._preprocess(observation)
            # calculate field differences
            if prev_field is None:
                prev_field = curr_field
                diff_field = curr_field
            else:
                diff_field = curr_field - prev_field
            state = np.concatenate([diff_field, player])

            # calculate probability gradient
            p_decision = to_categorical(action,
                                        num_classes=self.env.get_action_space().n)
            gradient = p_decision.astype('float32') - p_action

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
        d_rewards = np.zeros_like(rewards).astype('float32')
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
        opponent, curr_field, player = Agent_PG._preprocess(observation)
        # calculate field differences
        if self._prev_field is None:
            self._prev_field = curr_field
            diff_field = curr_field
        else:
            diff_field = curr_field - self._prev_field
        # concat every vectors as a single state
        state = np.concatenate([diff_field, player])
        # reverse the input to accomodate the requirement
        state = state.reshape([1, state.shape[0]])

        p_action = self.model.predict(state, batch_size=1).flatten()
        # normalize the PDF
        p_action /= np.sum(p_action)

        # determine the action according to the PDF
        action = np.random.choice(self.env.get_action_space().n, p=p_action)
        #action = np.argmax(p_action)
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
            opponent: np.array
                opponent status
            field: np.array
                field status, weighted by distance to the player
            player: np.array
                player status
        """
        # convert to grayscale
        observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        # binarize, {0, 1}
        _, observation = cv2.threshold(observation.astype(np.uint8), 0, 1,
                                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # convert back to float
        observation = observation.astype('float32')

        # opponent
        #   x=16, y=34, w=4, h=160
        opponent = observation[34:194, 19]
        opponent /= np.sum(opponent)

        # field
        #   x=20, y=34, w=120, h=160
        field = observation[34:194, 20:140]
        # apply weights (w length) for the field position
        weights = np.arange(1, 121).astype('float32') / 120
        field = np.dot(field, weights)

        # player
        #   x=140, y=34, w=4, h=160
        player = observation[34:194, 140]
        player /= np.sum(field)

        return opponent, field, player
