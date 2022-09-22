import os, sys
import numpy as np
sys.setrecursionlimit(1500)
os.system("clear")


## ###################
## HELPER FUNCTIONS
## ###################
def debug():
  import pdb
  pdb.set_trace()

def getIndexClosestValue(list_vals, target_val):
  ## work with arrays
  array_vals = np.asarray(list_vals)
  return np.argmin(np.abs(array_vals - target_val))

def createLogFile():
  with open("log.txt", "w") as txt_file:
    txt_file.write(f"Tic Tac Toe v2.0\n")
    txt_file.write(f"\n")

def appendToLogFile(str):
  with open("log.txt", "a") as txt_file:
    txt_file.write(f"{str}\n")


## ###########################
## PROGRAM PARAMETERS
## ###########################
BOOL_DEBUG    = 0
MAX_DEPTH     = 5
SYMBOL_PLAYER = "+"
SYMBOL_AI     = "-"


## ###################
## GAME CLASS
## ###################
class TicTacToe():
  def __init__(self):
    self.initialise()
    if BOOL_DEBUG:
      createLogFile()
      self.tests()

  def initialise(self):
    self.board = np.zeros((3,3))
    self.list_piece_sizes_p1 = [ 1,  2,  3,  4,  5]
    self.list_piece_sizes_p2 = [-1, -2, -3, -4, -5]
    self.list_piece_flags_p1 = [1] * len(self.list_piece_sizes_p1)
    self.list_piece_flags_p2 = [1] * len(self.list_piece_sizes_p2)

  def play(self):
    print(f"You are '{SYMBOL_PLAYER}', and your oponent (an AI) is '{SYMBOL_AI}'.")
    print(" ")
    depth = 0
    self.initialise()
    while True:
      self.printBoard()
      status = self.checkGameOverStatus()
      ## check if the game has concluded
      if status is not None:
        print("The game has concluded.")
        if status > 0:
          print("You are the winner!")
        elif status < 0:
          print("The AI is the winner!")
        else: print("It is a tie!")
        return
      ## player 1's (user) turn
      if depth % 2 == 0:
        if all((flag == 0) for flag in self.list_piece_flags_p1):
          print("You are out of pieces.")
          print(" ")
        else:
          ## get user's move
          (x, y), piece_index = self.getPlayerMove()
          self.board[y][x] = self.list_piece_sizes_p1[piece_index]
          self.list_piece_flags_p1[piece_index] = 0
          print(" ")
      ## player 2's (AI) turn
      else:
        if all((flag == 0) for flag in self.list_piece_flags_p2):
          print("AI is out of pieces.")
          print(" ")
        else:
          ## choose best move based on minimax algorithm
          _, (x, y), piece_index = self.max(-np.inf, np.inf, 0)
          piece_size = self.list_piece_sizes_p2[piece_index]
          self.board[y][x] = piece_size
          self.list_piece_flags_p2[piece_index] = 0
          print(f"The AI's move is: ({x}, {y}), size: {piece_size}.")
          print(" ")
      ## increment depth
      depth += 1

  def getPlayerMove(self):
    while True:
      print("Choose your move.")
      x = int(input("Column (x): "))
      y = int(input("   Row (y): "))
      if not self.isPlayerMoveValid((x, y)):
        print("Your chosen coordinate was invalid.")
        print(" ")
        continue
      piece_size = int(input("Size (1-5): "))
      piece_index = getIndexClosestValue(self.list_piece_sizes_p1, piece_size)
      msg_bool, msg_flag = self.isSizeValid(x, y, piece_index)
      if not msg_bool:
        print(f"Your chosen size was invalid. {msg_flag}")
        print(" ")
        continue
      return (x, y), piece_index

  def isPlayerMoveValid(self, to_coord):
    x, y = to_coord
    ## check x-coord lies within board bounds
    if (x < 0) or (2 < x):
      return False
    ## check y-coord lies within board bounds
    if (y < 0) or (2 < y):
      return False
    ## check that the target cell is empty or occupied by player 2 (the AI)
    return self.board[y][x] <= 0

  def isSizeValid(self, x, y, piece_index):
    piece_size = self.list_piece_sizes_p1[piece_index]
    ## check if the index is out of range
    if (piece_index < 0) or (len(self.list_piece_sizes_p1)-1 < piece_index):
      return False, 1.1
    ## check if the size is out of range
    if (piece_size < min(self.list_piece_sizes_p1)) or (max(self.list_piece_sizes_p1) < piece_size):
      return False, 1.2
    ## check if piece is available
    if self.list_piece_flags_p1[piece_index] == 0:
      return False, 2
    ## check that the player does not have a piece placed there
    if self.board[y][x] > 0:
      return False, 3
    ## check that the piece being placed has higher value
    if abs(self.board[y][x]) >= piece_size:
      return False, 4
    ## everything is okay
    return True, 0

  def max(self, alpha_score, beta_score, depth=0):
    ## check if the game has concluded
    status = self.checkGameOverStatus()
    if status is not None:
      if BOOL_DEBUG:
        self.printBoard()
        appendToLogFile(f"AI: End of search. Game status: {status}")
        appendToLogFile(" ")
      return -1*status, (None, None), None
    ## initialise to worst case
    max_score = -np.inf
    max_x = None
    max_y = None
    max_piece_index = None
    ## loop over possible moves
    for y_index in range(3):
      for x_index in range(3):
        for piece_index in [
            index for index, flag in enumerate(self.list_piece_flags_p2)
            if abs(flag) > 0
          ]:
          ## (spot is free or occupied by player) and (player's piece is smaller)
          if ((self.board[y_index][x_index] >= 0) and
              (abs(self.board[y_index][x_index]) < abs(self.list_piece_sizes_p2[piece_index]))):
            ## make temporary AI move
            if BOOL_DEBUG:
              appendToLogFile(f"AI: ({x_index}, {y_index}), p_old = {self.board[y_index][x_index]}, p_new = {self.list_piece_sizes_p2[piece_index]}")
            prev_piece = self.board[y_index][x_index]
            self.board[y_index][x_index] = self.list_piece_sizes_p2[piece_index]
            self.list_piece_flags_p2[piece_index] = 0
            ## determine the best move that the player can play
            score, _, _ = self.min(alpha_score, beta_score, depth+1)
            if (score > max_score):
              max_score = score
              max_x, max_y = x_index, y_index
              max_piece_index = piece_index
            ## reset: remove temporary AI move
            self.board[y_index][x_index] = prev_piece
            self.list_piece_flags_p2[piece_index] = 1
            ## something
            if (max_score >= beta_score) or (depth >= MAX_DEPTH):
              return max_score, (max_x, max_y), max_piece_index
            if max_score > alpha_score:
              alpha_score = max_score
    return max_score, (max_x, max_y), max_piece_index

  def min(self, alpha_score, beta_score, depth=0):
    ## check if the game has concluded
    status = self.checkGameOverStatus()
    if status is not None:
      if BOOL_DEBUG:
        self.printBoard()
        appendToLogFile(f"P1: End of search. Game status: {status}")
        appendToLogFile(" ")
      return -1*status, (None, None), None
    ## initialise to worst case
    min_score = np.inf
    min_x = None
    min_y = None
    min_piece_index = None
    ## loop over possible moves
    for y_index in range(3):
      for x_index in range(3):
        for piece_index in [
            index for index, flag in enumerate(self.list_piece_flags_p1)
            if abs(flag) > 0
          ]:
          ## (spot is free or occupied by AI) and (AI's piece is smaller)
          if ((self.board[y_index][x_index] <= 0) and
              (abs(self.board[y_index][x_index]) < abs(self.list_piece_sizes_p1[piece_index]))):
            ## make temporary player move
            if BOOL_DEBUG:
              appendToLogFile(f"P1: ({y_index}, {x_index}), p_old = {self.board[y_index][x_index]}, p_new = {self.list_piece_sizes_p1[piece_index]}")
            prev_piece = self.board[y_index][x_index]
            self.board[y_index][x_index] = self.list_piece_sizes_p1[piece_index]
            self.list_piece_flags_p1[piece_index] = 0
            ## determine the best move that the AI can play
            score, _, _ = self.max(alpha_score, beta_score, depth+1)
            if (score < min_score):
              min_score = score
              min_x, min_y = x_index, y_index
              min_piece_index = piece_index
            ## reset: remove temporary player move
            self.board[y_index][x_index] = prev_piece
            self.list_piece_flags_p1[piece_index] = 1
            ## something
            if (min_score <= alpha_score) or (depth >= MAX_DEPTH):
              return min_score, (min_x, min_y), min_piece_index
            if min_score < beta_score:
              beta_score = min_score
    return min_score, (min_x, min_y), min_piece_index

  def checkGameOverStatus(self):
    ## check for wins: player 1 and 2, respectively
    for player_sgn in [1, -1]:
      ## check if a horizontal win occured
      for y_index in range(3):
        if ((player_sgn * self.board[y_index][0] > 0) and 
            (player_sgn * self.board[y_index][1] > 0) and
            (player_sgn * self.board[y_index][2] > 0)):
          return player_sgn
      ## check if a vertical win occured
      for x_index in range(3):
        if ((player_sgn * self.board[0][x_index] > 0) and
            (player_sgn * self.board[1][x_index] > 0) and
            (player_sgn * self.board[2][x_index] > 0)):
          return player_sgn
      ## check if a top-left to bottom-right diagonal win soccured
      if ((player_sgn * self.board[0][0] > 0) and
          (player_sgn * self.board[1][1] > 0) and
          (player_sgn * self.board[2][2] > 0)):
        return player_sgn
      ## check if a top-right to bottom-left diagonal win occured
      if ((player_sgn * self.board[0][2] > 0) and
          (player_sgn * self.board[1][1] > 0) and
          (player_sgn * self.board[2][0] > 0)):
        return player_sgn
    ## check if any pieces are still remaining
    if any(self.list_piece_flags_p1) or any(self.list_piece_flags_p2):
      return None
    ## it is a tie
    return 0

  def printBoard(self):
    print("x:", "   ".join(["0", " 1", " 2"]), "| y")
    print("-" * (5*3 + 3))
    for y_index, row in enumerate(self.board):
      for cell in row:
        s = int(abs(cell))
        if   cell > 0: p = SYMBOL_PLAYER
        elif cell < 0: p = SYMBOL_AI
        else: p = s = "*"
        print(f"| {p}{s}", end=" ")
      print(f"| {y_index}")
    print("Player 1's pieces:", [
      f"{SYMBOL_PLAYER}{abs(piece_size)}"
      for piece_index, piece_size in enumerate(self.list_piece_sizes_p1)
      if self.list_piece_flags_p1[piece_index] > 0
    ])
    print("Player 2's pieces:", [
      f"{SYMBOL_AI}{abs(piece_size)}" for piece_index, piece_size in enumerate(self.list_piece_sizes_p2)
      if self.list_piece_flags_p2[piece_index] > 0
    ])
    print(" ")
    if BOOL_DEBUG:
      appendToLogFile(self.board)
      appendToLogFile("Player 1's pieces:")
      appendToLogFile([
        f"x{abs(piece_size)}"
        for piece_index, piece_size in enumerate(self.list_piece_sizes_p1)
        if self.list_piece_flags_p1[piece_index] > 0
      ])
      appendToLogFile("Player 2's pieces:")
      appendToLogFile([
        f"o{abs(piece_size)}" for piece_index, piece_size in enumerate(self.list_piece_sizes_p2)
        if self.list_piece_flags_p2[piece_index] > 0
      ])

  def tests(self):
    ## test: draw/index board correctly
    list_board = [ 3, 1, -5, 4, 5, 0, -2, 0, -4]
    self.board = np.array([
      [ 3,  1, -5],
      [ 4,  5,  0],
      [-2,  0, -4]
    ])
    elem_index = 0
    for y_index in range(3):
      for x_index in range(3):
        if (self.board[y_index][x_index] != list_board[elem_index]):
          raise Exception(f"Failed test: print board. Info: {self.board[y_index][x_index]}, {list_board[elem_index]}")
        elem_index += 1
    ## test: detect free pieces
    self.list_piece_flags_p1 = [0, 0, 0, 0, 0]
    self.list_piece_flags_p2 = [0, 0, 0, 0, 1]
    if self.checkGameOverStatus() is not None:
      raise Exception("Failed test: detect free pieces.")
    ## check tests work for both players
    for val in [1, -1]:
      ## test: diagonal win (top-left to bottom-right)
      self.board = np.array([
        [val, 0, 0],
        [0, val, 0],
        [0, 0, val]
      ])
      if not (self.checkGameOverStatus() == val):
        raise Exception(f"Failed test: detect diagonal win. Info: {val}")
      ## test: off-diagonal win (top-right to bottom-left)
      self.board = np.array([
        [0, 0, val],
        [0, val, 0],
        [val, 0, 0]
      ])
      if not (self.checkGameOverStatus() == val):
        raise Exception(f"Failed test: detect off-diagonal win. Info: {val}")
      ## test: row win
      for x_index in range(3):
        self.board = np.zeros((3,3))
        self.board[:, x_index] = val
        if not (self.checkGameOverStatus() == val):
          raise Exception(f"Failed test: detect row win. Info: {x_index} {val}")
      ## test: column win
      for y_index in range(3):
        self.board = np.zeros((3,3))
        self.board[y_index, :] = val
        if not (self.checkGameOverStatus() == val):
          raise Exception(f"Failed test: detect column win. Info: {y_index} {val}")
    ## test: detect drawn game
    self.board = np.array([
      [ 3,  1, -5],
      [ 4,  5,  0],
      [-2,  0, -4]
    ])
    self.list_piece_flags_p1 = [0, 0, 0, 0, 0]
    self.list_piece_flags_p2 = [0, 0, 0, 0, 0]
    result = self.checkGameOverStatus()
    if (result != 0):
      raise Exception(f"Failed test 6: detect drawn position. Info: {result}")
    ## test: AI makes correct move
    self.list_piece_flags_p1 = [0, 0, 1, 1, 1]
    self.list_piece_flags_p2 = [0, 1, 1, 1, 1]
    self.board = np.array([
      [-1,  1,  0],
      [ 0,  2,  0],
      [ 0,  0,  0]
    ])
    _, (x, y), piece_index = self.max(-np.inf, np.inf)
    piece_size = self.list_piece_sizes_p2[piece_index]
    self.board[y][x] = piece_size
    self.list_piece_flags_p2[piece_index] = 0
    bool_good_move_1 = (x == 1) and (y == 0)
    bool_good_move_2 = (x == 1) and (y == 1)
    bool_good_move_3 = (x == 1) and (y == 2)
    if BOOL_DEBUG:
      self.printBoard()
      print(f"The AI's move is: ({x}, {y}), size: {piece_size}.")
      print(" ")
      self.printBoard()
    if not ((bool_good_move_1 or bool_good_move_2) or (bool_good_move_3)):
      raise Exception(f"Failed test 2: make best move. (x, y)=({x}, {y}) and s={self.list_piece_sizes_p2[piece_index]}")
    ## success
    print("Passed all tests.")
    print(" ")


## ###################
## DEFINE MAIN PROGRAM
## ###################
def main():
  game = TicTacToe()
  game.play()


## ###################
## RUN MAIN
## ###################
if __name__ == "__main__":
  main()


## END OF PROGRAM