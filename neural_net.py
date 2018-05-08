import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pickle
import collections

WHITE, BLACK, CORNER, BLANK, REMOVED = ['O','@','X','-',' ']
ENEMIES = {WHITE: {BLACK, CORNER}, BLACK: {WHITE, CORNER}}
FRIENDS = {WHITE: {WHITE, CORNER}, BLACK: {BLACK, CORNER}}
BOARD_SIZE = 8
MAX_TURNS = 300
MAX_PIECES = 12
EDGE1 = [0, BOARD_SIZE - 1]
EDGE2 = [1, BOARD_SIZE - 2]
CORNERS1 = [(1, 1), (1, 6), (6, 6), (6, 1)]
CORNERS2 = [(2, 2), (2, 5), (5, 5), (5, 2)]

class NeuralNet:

    def __init__(self):
        # hyperparameters
        self.episode_number = 0
        self.batch_size = 10
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99
        self.hidden = 70
        self.input = 145
        self.learning_rate = 1
        self.reward_sum = 0
        self.myLambda = 0.7
        self.running_reward = None
        self.prev_processed_observations = None
        self.weights = {}
        self.tdLeaf = []

        try:
            self.weights = pickle.load(open("weights.p", "rb"))

        except:
            self.weights["w1"] = np.random.randn(self.hidden, self.input) / np.sqrt(self.input)
            self.weights["w2"] = np.random.randn(self.hidden)/np.sqrt(self.hidden)
            print(self.weights)

        pickle.dump(self.weights, open("weights.p", "wb"))


    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def relu(self, x):
        if x < 0:
            return 0
        return x

    def prepro(self, board, colour, turns):
        vector =[]
        if colour == BLACK:
            friendly_pieces = board.black_pieces
            enemy_pieces = board.white_pieces
        else:
            friendly_pieces = board.white_pieces
            enemy_pieces = board.black_pieces

        # number of turns
        vector.append(turns/MAX_TURNS)

        # no. friendly pieces, no. enemy pieces

        vector.append(len(friendly_pieces)/MAX_PIECES)
        vector.append(len(enemy_pieces)/MAX_PIECES)

        #placing phase
        if turns < 24:
            vector.append(1)
        else:
            vector.append(0)

        #moving phase 1
        if 24 <= turns < 128:
            vector.append(1)
        else:
            vector.append(0)

        #moving phase 2
        if 128 <= turns <192:
            vector.append(1)
        else:
            vector.append(0)

        #moving phase 3
        if turns >= 192:
            vector.append(1)
        else:
            vector.append(0)

        #Attack defend map
        grid = {}


        for piece in friendly_pieces:
            for _, pos in piece.moves():
                if pos in grid:
                    grid[pos] += 1
                else:
                    grid[pos] = 1

        for piece in enemy_pieces:
            for _, pos in piece.moves():
                if pos in grid:
                    grid[pos] += -1
                else:
                    grid[pos] = -1


        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if (x, y) not in grid:
                    grid[(x, y)] = 0
                vector.append(grid[(x, y)])

        #friendly and enemy pieces on each grid
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                piece = board.grid[(x, y)]
                if piece in ENEMIES[colour]:
                    vector.append(-1)
                elif piece == colour:
                    vector.append(1)
                else:
                    vector.append(0)

        #pieces in outer edge
        friendlies = 0
        for piece in friendly_pieces:
            x, y = piece.pos
            if x in EDGE1 or y in EDGE1:
                friendlies+=1
        vector.append(friendlies)
        enemies = 0
        for piece in enemy_pieces:
            x, y = piece.pos
            if x in EDGE1 or y in EDGE1:
                enemies += 1
        vector.append(enemies)

        # pieces in second edge
        friendlies = 0
        for piece in friendly_pieces:
            x, y = piece.pos
            if x in EDGE2 or y in EDGE1:
                friendlies += 1
        vector.append(friendlies)
        enemies = 0
        for piece in enemy_pieces:
            x, y = piece.pos
            if x in EDGE2 or y in EDGE1:
                enemies += 1
        vector.append(enemies)

        #pieces in corners 1
        friendlies = 0
        for piece in friendly_pieces:
            if piece.pos in CORNERS1:
                friendlies += 1
        vector.append(friendlies)
        enemies = 0
        for piece in enemy_pieces:
            if piece.pos in CORNERS1:
                enemies += 1
        vector.append(enemies)

        # pieces in corners 2
        friendlies = 0
        for piece in friendly_pieces:
            if piece.pos in CORNERS2:
                friendlies += 1
        vector.append(friendlies)
        enemies = 0
        for piece in enemy_pieces:
            if piece.pos in CORNERS2:
                enemies += 1
        vector.append(enemies)

        #friendly moves
        friendly_moves = 0
        for piece in friendly_pieces:
            friendly_moves += len(piece.moves())
        vector.append(friendly_moves)

        # enemy moves
        enemy_moves = 0
        for piece in enemy_pieces:
            enemy_moves += len(piece.moves())
        vector.append(friendly_moves)
        return vector

    def evaluateBoardAdvanced(self, board, colour, turns):
        #self.weights = pickle.load(open("weights.p", "rb"))
        observations = self.prepro(board, colour, turns)
        layer1 = np.dot(self.weights["w1"], observations)
        layer1_activated = list(map(self.relu, layer1))
        layer2 = np.dot(self.weights["w2"], layer1_activated)
        #print(self.sigmoid(layer2))
        return self.sigmoid(layer2), layer2, layer1

    def deriv_sigmoid(self, x):
        return np.exp(-x) / ((1.0 + np.exp(-x))**2)

    def deriv_relu(self, x):
        if x <= 0:
            return 0
        return 1

    def updateWeight1(self, weight, index):
        term1 = 0
        for x in range(1, len(self.tdLeaf)):
            term2 = 0
            for y in range(1, len(self.tdLeaf)):
                term2 += (self.myLambda ** (y - x)) * (self.tdLeaf[y][0] - self.tdLeaf[y - 1][0])
            term1 += term2 * self.deriv_sigmoid(self.tdLeaf[x-1][1]) * self.weights["w2"][index] * self.deriv_relu(self.tdLeaf[x-1][2][index] * weight)
        new_weight = weight + term1 * self.learning_rate
        return new_weight


    def updateWeight2(self,weight):
        term1 = 0
        for x in range(1, len(self.tdLeaf)):
            term2 = 0
            for y in range(1, len(self.tdLeaf)):
                term2 += (self.myLambda ** (y - x)) * (self.tdLeaf[y][0] - self.tdLeaf[y - 1][0])
            term1 += term2 * self.deriv_sigmoid(self.tdLeaf[x-1][1]) * weight
        new_weight = weight + term1 * self.learning_rate
        return new_weight

    def updateWeights(self):
        self.tdLeaf = pickle.load(open("tdLeaf.p", "rb"))
        for index in range(self.hidden):
            self.weights["w1"][index] = np.array([self.updateWeight1(x, index) for x in self.weights["w1"][index]])
        self.weights["w2"] = np.array(list(map(self.updateWeight2, self.weights["w2"])))
        self.weights["w2"] = self.weights["w2"]
        #print(self.weights)
        pickle.dump(self.weights, open("weights.p", "wb"))
        return

