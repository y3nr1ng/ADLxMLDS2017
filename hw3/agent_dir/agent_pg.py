from agent_dir.agent import Agent

import numpy as np
import pickle as pickle

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

D = 80 * 80 # input dimensionality: 80x80 grid

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        if args.test_pg or args.reuse:
            model = pickle.load(open('agent_pg_weight.p', 'rb'))
        else:
            model = {}
            model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
            model['W2'] = np.random.randn(H) / np.sqrt(H)

        self.model = model
        self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

        self.prev_x = None

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
        pass

    def discount_rewards(r):
        """
        Take 1D float array of rewards and compute discounted reward.
        """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_backward(self, eph, epdlogp):
        """
        Backward pass. (eph is array of intermediate hidden states)

        Parameters
        ----------
        eph: np.array
            array of intermediate hidden states
        epdlogp: np.array
            discounted log(P) gradients

        Returns
        -------
        Updated hidden states.
        """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent.
        """
        # preprocess the observation, set input to network to be difference image
        cur_x = self.prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(D)
        self.prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = self.policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        if test:
            return action
        else:
            return action, aprob

    def prepro(self, I):
        """
        Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
        """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state
