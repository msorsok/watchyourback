import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pickle

WHITE, BLACK, CORNER, BLANK, REMOVED = ['O','@','X','-',' ']
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}}
BOARD_SIZE = 8

#np.random.seed(42)

class NeuralNet:

    def __init__(self):
        # hyperparameters
        self.episode_number = 0
        self.batch_size = 10
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99
        self.hidden = 32
        self.input = 64
        self.learning_rate = 0.1
        self.reward_sum = 0
        self.myLambda = 0.7
        self.running_reward = None
        self.prev_processed_observations = None
        self.weights = {}

        try:
            self.weights = pickle.load(open("weights.p.p", "rb"))

        except:
            self.weights["w1"] = np.random.random_sample((self.hidden, self.input))
            print(self.weights["w1"])
            self.weights["w2"] = np.random.random_sample(self.hidden)
            self.weights["w2"] = self.weights["w2"] / np.amax(np.abs(self.weights["w2"]))
            print(self.weights["w2"])

        pickle.dump(self.weights, open("weights.p.p", "wb"))

        self.batch_observations = []
        self.batch_layer1 = []
        self.batch_layer2 = []


    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def relu(self, x):
        if x < 0:
            return 0
        return x

    def prepro(self, board, colour):
        #returns 1D vector of board pieces
        vector =[]
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                piece = board.grid[(x,y)]
                if piece in ENEMIES[colour]:
                    vector.append(-1)
                elif piece == colour:
                    vector.append(1)
                else:
                    vector.append(0)
        return vector

    def evaluateBoardAdvanced(self, board, colour):
        self.weights = pickle.load(open("weights.p.p", "rb"))
        observations = self.prepro(board, colour)
        layer1 = np.dot(self.weights["w1"], observations)
        layer1_activated = list(map(self.relu, layer1))
        layer2 = np.dot(self.weights["w2"], layer1_activated)
        return self.sigmoid(layer2), layer2, layer1

    def deriv_sigmoid(self, x):
        return np.exp(-x) / ((1.0 + np.exp(-x))**2)

    def deriv_relu(self, x):
        if x <= 0:
            return 0
        return 1



    def updateWeight1(self, weight, index):
        tdLeaf = pickle.load(open("tdLeaf.p", "rb"))
        term1 = 0
        for x in range(1, len(tdLeaf)):
            term2 = 0
            for y in range(1, len(tdLeaf)):
                term2 += (self.myLambda ** (y - x)) * (tdLeaf[y][0] - tdLeaf[y - 1][0])
            term1 += term2 * self.deriv_sigmoid(tdLeaf[x-1][1]) * self.weights["w2"][index] * self.deriv_relu(tdLeaf[x-1][2][index] * weight)
        new_weight = weight + term1 * self.learning_rate
        return new_weight


    def updateWeight2(self,weight):
        tdLeaf = pickle.load(open("tdLeaf.p", "rb"))
        term1 = 0
        for x in range(1, len(tdLeaf)):
            term2 = 0
            for y in range(1, len(tdLeaf)):
                term2 += (self.myLambda ** (y - x)) * (tdLeaf[y][0] - tdLeaf[y - 1][0])
            term1 += term2 * self.deriv_sigmoid(tdLeaf[x-1][1]) * weight
        new_weight = weight + term1 * self.learning_rate
        return new_weight

    def updateWeights(self):
        for index in range(len(self.weights["w2"])):
            self.weights["w1"][index] = np.array([self.updateWeight1(x, index) for x in self.weights["w1"][index]])

        self.weights["w2"] = np.array(list(map(self.updateWeight, self.weights["w2"])))
        self.weights["w2"] = self.weights["w2"] / np.amax(np.abs(self.weights["w2"]))
        pickle.dump(self.weights, open("weights.p.p", "wb"))
        return

