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
        #self.weights["w1"] = np.random.randn(self.hidden, self.input) / np.sqrt(self.input)
        try:
            self.weights["w2"] = pickle.load(open("weights", "rb"))
            self.weights["w2"] = self.weights["w2"] / np.amax(np.abs(self.weights["w2"]))
        except:
            print("no weights found, making new")
            self.weights["w2"] = np.random.random_sample(self.input)
            pickle.dump(self.weights["w2"], open("weights", "wb"))
            self.weights["w2"] = self.weights["w2"] / np.amax(np.abs(self.weights["w2"]))

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
        self.weights["w2"] = pickle.load(open("weights", "rb"))
        observations = self.prepro(board, colour)
        #layer1 = np.dot(self.weights[0], observations)
        #layer1 = list(map(self.relu, layer1))
        #self.batch_layer1.append(layer1)
        layer2 = np.dot(self.weights["w2"], observations)
        return self.sigmoid(layer2), layer2

    def deriv_sigmoid(self, x):
        return np.exp(-x) / ((1.0 + np.exp(-x))**2)

    def updateWeight(self,weight):
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
        self.weights["w2"] = np.array(list(map(self.updateWeight, self.weights["w2"])))
        self.weights["w2"] = self.weights["w2"] / np.amax(np.abs(self.weights["w2"]))
        pickle.dump(self.weights["w2"], open("weights", "wb"))
        return

