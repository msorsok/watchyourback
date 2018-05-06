
from state_representation import *
from neural_net_1layer import NeuralNet
# initialise player
# set up internal representation of the board
# colour is either 'white' or 'black'

import pickle

BOARD_SIZE = 8
PLACING_ACTIONS = 12
SHRINK_ACTIONS = [75, 107]
SHRINK_BEFORE = [128, 192]
SHRINK_AFTER = [127, 191]
SHRINK1 = [127, 128]
SHRINK2 = [191, 192]

class Player:

    def __init__(self, colour):
        self.actions = 0
        self.board = self.initialiseBoard()
        self.neuralNet = NeuralNet()
        self.tdLeaf = []

        if colour.upper() == "BLACK":
            self.colour = BLACK
        else:
            self.colour = WHITE


    """actions a player can make -
        make place a piece on a square, use tuple (x,y)
        move a piece from one square to another, use nested tuple ((a,b), (c,d))
        forfeit their turn, use value None """

    # player selects next action and returns it given current state of the board
    # turns - number of turns that have taken place since start of current game phase
    def action(self, turns):
        if turns in SHRINK_BEFORE:
            self.shrink(turns)

        if self.actions < PLACING_ACTIONS:
            actions = self.generatePlacingActions()

        else:
            actions = self.generateMovingActions()

        bestProb = 0
        bestAction = None
        bestScore = 0

        if actions:
            for action in actions:
                curr_prob, score = self.evaluateAction(action)
                if curr_prob > bestProb:
                    bestAction = action
                    bestProb = curr_prob
                    bestScore = score

            self.updateBoard(bestAction, self.colour)
        self.actions += 1

        if turns in SHRINK_AFTER:
            self.shrink(turns)
        return bestAction


    # update player about opponents moves
    def update(self, action):
        if self.colour == BLACK:
            enemyColour = WHITE
        else:
            enemyColour = BLACK
        self.updateBoard(action, enemyColour)
        return


    #returns a new board object with no pieces on it
    def initialiseBoard(self):
        layout = [ [ BLANK for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for x,y in [(0,0),(0,BOARD_SIZE -1),(BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]:
            layout[x][y] = CORNER
        return Board(layout)


    #returns a list of all possible actions in placing phase
    def generatePlacingActions(self):
        if self.colour == WHITE:
            y_start = 0
            y_end = 6
        else:
            y_start = 2
            y_end = BOARD_SIZE

        actions = []
        for x in range(BOARD_SIZE):
            for y in range(y_start, y_end):
                if self.board.grid[(x,y)] == BLANK:
                    actions.append((x,y))
        return actions

    #returns a list of all possible actions in moving phase
    def generateMovingActions(self):
        actions = []
        if self.colour == BLACK:
            pieces = self.board.black_pieces
        else:
            pieces = self.board.white_pieces

        for piece in pieces:
            actions += piece.moves()
        return actions

    #evaluates the probability of winning if a certain action is taken
    def evaluateAction(self, action):

        x, y = action
        #if placing action
        if (isinstance(x, int) and isinstance(y, int)):
            piece = Piece(self.colour, (x,y), self.board)
            eliminated = piece.makemove((x,y))
            prob, score = self.neuralNet.evaluateBoardAdvanced(self.board, self.colour)
            piece.undomove((x, y), eliminated)
            piece.eliminate()

        #if moving action
        else:
            piece = self.board.find_piece(x)
            eliminated = piece.makemove(y)
            prob, score = self.neuralNet.evaluateBoardAdvanced(self.board, self.colour)
            piece.undomove(x, eliminated)

        return prob, score

    #enacts action on board
    def updateBoard(self, action, colour):
        if action:
            x, y = action
            if (isinstance(x, int) and isinstance(y, int)):
                piece = Piece(colour, (x, y), self.board)
                piece.makemove((x, y))
            else:
                piece = self.board.find_piece(x)
                piece.makemove(y)
            return


    def shrink(self, turns):
        if turns in SHRINK1:
            edge = [0, BOARD_SIZE - 1]
            corners = [(1, 1), (1, 6), (6, 6), (6, 1)]
        elif turns in SHRINK2:
            edge = [1, BOARD_SIZE - 2]
            corners = [(2, 2), (2, 5), (5, 5), (5, 2)]
        else:
            print("shrinking gone wrong")

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if x in edge or y in edge:
                    piece = self.board.find_piece((x,y))
                    if piece:
                        piece.eliminate()
                    self.board.grid[(x,y)] = REMOVED

        for corner in corners:
            piece = self.board.find_piece(corner)
            if piece:
                piece.eliminate()
            self.board.grid[corner] = CORNER
            for direction in DIRECTIONS:
                adjacent_square = tuple(map(sum, zip(corner, direction)))
                opposite_square = tuple(map(sum, zip(adjacent_square, direction)))
                adjacent_piece = self.board.find_piece(adjacent_square)
                opposite_piece = self.board.find_piece(opposite_square)
                if adjacent_piece and opposite_piece:
                    if adjacent_piece.player != opposite_piece.player:
                        adjacent_piece.eliminate()
        return
