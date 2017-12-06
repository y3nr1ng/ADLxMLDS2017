from agent_dir.agent import Agent

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import print_summary

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        self._build_network()
        if args.test_pg:
            self.model.load_weights('agent_pg_weight.h5')
        self._compile_network()

        raise RuntimeError('DEBUG')

    def _build_network(self):
        """
        Create a base network.
        """
        # parse network size
        #   in: 3 types of input, (opponent, field, player)
        #   out: n actions
        in_dim = 160 * 3
        out_dim = self.env.get_action_space().shape[0]

        model = Sequential([
            Dense(128, input_shape=(in_dim, )),
            Activation('relu'),
            Dense(out_dim),
            Activation('softmax')
        ])

        print_summary(model)
        self.model = model

    def _compile_network(self):
        self.model.compile(optimiazer='adam',
                           loss='categorical_crossentropy',
                           metrics='accuracy')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """


        self.model.save_weights('agent_pg_weight.h5')

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
        # convert to grayscale
        observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        # binarize, {0, 1}
        _, observation = cv2.threshold(observation.astype(np.uint8), 0, 1,
                                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # field
        #   x=20, y=34, w=120, h=160
        field = observation[34:194, 20:140]
        # player
        #   x=140, y=34, w=4, h=160
        player = observation[34:194, 140]
        # opponent
        #   x=16, y=34, w=4, h=160
        opponent = observation[34:194, 19]

        # apply weights (w length) for the field position
        weights = np.arange(1, 121)
        field = np.dot(field, weights)

        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()
