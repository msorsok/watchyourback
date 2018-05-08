from state_representation import *
from neural_net import NeuralNet
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
        self.isPlacing = True
        self.board = self.initialiseBoard()
        self.neuralNet = NeuralNet()
        self.tdLeaf = []
        self.actions = 0

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

        if self.actions != 0 and (self.actions % 12 == 0):
            print("updating weights")
            self.neuralNet.updateWeights()
            self.tdLeaf = []

        action, prob, score, hiddenScore, alpha = self.max_value(2, None, self.colour, turns, self.isPlacing, None, None)

        if turns in SHRINK_BEFORE:
            self.shrink(turns)

        self.updateBoard(action, self.colour)

        if turns in SHRINK_AFTER:
            self.shrink(turns)

        if self.isPlacing and turns == 23:
            self.isPlacing = False

        if self.isPlacing and turns == 22:
            self.isPlacing = False

        self.actions += 1
        self.tdLeaf.append((prob, score, hiddenScore))
        pickle.dump(self.tdLeaf, open("tdLeaf.p", "wb"))

        return action


    # update player about opponents moves
    def update(self, action):
        if self.colour == BLACK:
            enemyColour = WHITE
        else:
            enemyColour = BLACK
        self.updateBoard(action, enemyColour)
        return


    def max_value(self, depth, action, colour, turns, isPlacing, alpha, beta):
        if depth == 0:
            prob, score, hiddenScore = self.evaluateAction(action, turns - 1, colour)
            return action, prob, score, hiddenScore, prob
        old_board = self.board
        update_information = self.updateBoard(action, colour)
        if turns in SHRINK_BEFORE:
            eliminated = self.shrink(turns)
        if isPlacing and turns == 24:
            isPlacing = False
            turns = -1
        if colour == BLACK:
            new_colour = WHITE
        else:
            new_colour = BLACK
        potential_actions = self.generateActions(isPlacing, colour)
        best_move, best_prob, best_score, best_hiddenScore = None, -1, None, None
        for move in potential_actions:
            temp_action, temp_prob, temp_score, temp_hiddenScore, temp_alphabeta = self.min_value(depth - 1, move, new_colour, turns + 1, isPlacing, alpha, beta)

            if temp_prob > best_prob:
                best_move, best_prob, best_score, best_hiddenScore = move, temp_prob, temp_score, temp_hiddenScore

            if alpha == None or temp_alphabeta > alpha:
                alpha = temp_alphabeta

            if beta != None and alpha != None:
                if alpha >= beta:
                    if turns in SHRINK_BEFORE:
                        self.unshrink(turns, eliminated)
                    self.undo_update(update_information)
                    return best_move, best_prob, best_score, best_hiddenScore, beta
        if turns in SHRINK_BEFORE:
            self.unshrink(turns, eliminated)
        self.undo_update(update_information)
        return best_move, best_prob, best_score, best_hiddenScore, alpha


    def min_value(self, depth, action, colour, turns, isPlacing, alpha, beta):
        if depth == 0:
            prob, score, hiddenScore = self.evaluateAction(action, turns, colour)
            return action, prob, score, hiddenScore, prob
        update_information = self.updateBoard(action, colour)
        if turns in SHRINK_BEFORE:
            eliminated = self.shrink(turns)

        if isPlacing and turns == 24:
            isPlacing = False
            turns = -1
        if colour == BLACK:
            new_colour = WHITE
        else:
            new_colour = BLACK
        best_move, best_prob, best_score, best_hiddenScore = None, 2, None, None
        for move in self.generateActions(isPlacing, colour):
            temp_action, temp_prob, temp_score, temp_hiddenScore, temp_alphabeta = self.max_value(depth - 1, move, new_colour, turns + 1, isPlacing, alpha, beta)

            if temp_prob < best_prob:
                best_move, best_prob, best_score, best_hiddenScore = move, temp_prob, temp_score, temp_hiddenScore

            if beta == None or temp_alphabeta < beta:
                beta = temp_alphabeta

            if beta != None and alpha != None:
                if alpha >= beta:
                    if turns in SHRINK_BEFORE:
                        self.unshrink(turns, eliminated)
                    self.undo_update(update_information)
                    return best_move, best_prob, best_score, best_hiddenScore, alpha
        if turns in SHRINK_BEFORE:
            self.unshrink(turns, eliminated)
        self.undo_update(update_information)
        return best_move, best_prob, best_score, best_hiddenScore, beta


    #returns a new board object with no pieces on it
    def initialiseBoard(self):
        layout = [ [ BLANK for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for x,y in [(0,0),(0,BOARD_SIZE -1),(BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]:
            layout[x][y] = CORNER
        return Board(layout)

    def generateActions(self, isPlacing, colour):
        if isPlacing:
            return self.generatePlacingActions(colour)
        return self.generateMovingActions(colour)

    #returns a list of all possible actions in placing phase
    def generatePlacingActions(self, colour):
        if colour == WHITE:
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
    def generateMovingActions(self, colour):
        actions = []
        if colour == BLACK:
            pieces = self.board.black_pieces
        else:
            pieces = self.board.white_pieces

        for piece in pieces:
            actions += piece.moves()
        if len(actions) == 0:
            return [None]
        return actions

    #evaluates the probability of winning if a certain action is taken
    def evaluateAction(self, action, turns, colour):
        if action == None:
            return self.neuralNet.evaluateBoardAdvanced(self.board, colour, turns)
        x, y = action
        #if placing action
        if (isinstance(x, int) and isinstance(y, int)):
            piece = Piece(colour, (x,y), self.board)
            eliminated = piece.makemove((x,y))
            prob, score, hiddenScore = self.neuralNet.evaluateBoardAdvanced(self.board, colour, turns)
            piece.undomove((x, y), eliminated)
            piece.eliminate()

        #if moving action
        else:
            piece = self.board.find_piece(x)
            eliminated = piece.makemove(y)
            prob, score, hiddenScore = self.neuralNet.evaluateBoardAdvanced(self.board, colour, turns)
            piece.undomove(x, eliminated)

        return prob, score, hiddenScore

    #enacts action on board
    def updateBoard(self, action, colour):
        if action != None:
            x, y = action
            if (isinstance(x, int) and isinstance(y, int)):
                piece = Piece(colour, (x, y), self.board)
                eliminated = piece.makemove((x, y))
            else:
                piece = self.board.find_piece(x)
                eliminated = piece.makemove(y)
            return piece, eliminated, action
        return None, None, None


    def undo_update(self, update_information):
        piece, eliminated, action = update_information
        if action:
            x, y = action
            if (isinstance(x, int) and isinstance(y, int)):
                piece.undomove((x, y), eliminated)
                piece.eliminate()
            else:
                piece.undomove(x, eliminated)

            return

    def unshrink(self, turns, eliminated):
        if turns in SHRINK1:
            edge = [0, BOARD_SIZE - 1]
            new_corners = [(0, 0), (0, 7), (7, 7), (7, 0)]
            old_corners = [(1, 1), (1, 6), (6, 6), (6, 1)]
        elif turns in SHRINK2:
            edge = [1, BOARD_SIZE - 2]
            new_corners = [(1, 1), (1, 6), (6, 6), (6, 1)]
            old_corners = [(2, 2), (2, 5), (5, 5), (5, 2)]
        else:
            print("shrinking gone wrong")

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if x in edge or y in edge:
                    self.board.grid[(x,y)] = BLANK

        for corner in new_corners:
            self.board.grid[corner] = CORNER
        for corner in old_corners:
            self.board.grid[corner] = BLANK
        for piece in eliminated:
            piece.resurrect()
        return

    def shrink(self, turns):
        eliminated = []
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
                        eliminated.append(piece)
                        piece.eliminate()
                    self.board.grid[(x,y)] = REMOVED

        for corner in corners:
            piece = self.board.find_piece(corner)
            if piece:
                eliminated.append(piece)
                piece.eliminate()
            self.board.grid[corner] = CORNER
            for direction in DIRECTIONS:
                adjacent_square = tuple(map(sum, zip(corner, direction)))
                opposite_square = tuple(map(sum, zip(adjacent_square, direction)))
                adjacent_piece = self.board.find_piece(adjacent_square)
                opposite_piece = self.board.find_piece(opposite_square)
                if adjacent_piece and opposite_piece:
                    if adjacent_piece.player != opposite_piece.player:
                        eliminated.append(adjacent_piece)
                        adjacent_piece.eliminate()
        return eliminated