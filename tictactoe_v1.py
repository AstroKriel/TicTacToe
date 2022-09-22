import os, time
import numpy as np
os.system("clear")


## ###########################
## PROGRAM PARAMETERS
## ###########################
BOOL_ALPHA_BETA = 1


## ###########################
## GAME CLASS
## ###########################
class TicTacToe():
  def __init__(self):
    self.tests()
    self.initialise()

  def initialise(self):
    self.depth = 0
    self.board = np.zeros((3,3))

  def play(self):
    print("You are 'x', and your oponent (an AI) is 'o'.")
    print(" ")
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
      ## player's turn
      if self.depth % 2 == 0:
        ## get the player's move
        while True:
          print("Choose your move.")
          x = int(input("Column (x): "))
          y = int(input("   Row (y): "))
          if self.isValidMove((x, y)):
            self.board[y][x] = 1
            print(" ")
            break
          else:
            print("Your move was invalid.")
            print(" ")
      ## AI's turn
      else:
        ## choose best move based on minimax algorithm
        t_start = time.time()
        _, (x, y) = self.max(-np.inf, np.inf)
        t_end = time.time()
        t_elapse = round(t_end - t_start, 7)
        self.board[x][y] = -1
        print(f"Evaluation time: {t_elapse} seconds.")
        print(f"The AI's move is: ({y}, {x})")
        print(" ")
      ## increment depth
      self.depth += 1

  def max(self, alpha, beta):
    ## check if the game has concluded
    status = self.checkGameOverStatus()
    if status is not None:
      return -1*status, (None, None)
    ## initialise to worst case
    max_score = -np.inf
    max_x = None
    max_y = None
    ## loop over possible moves
    for row_index in range(3):
      for col_index in range(3):
        ## free spot on the board
        if self.board[row_index][col_index] == 0:
          ## make temporary (AI) move
          self.board[row_index][col_index] = -1
          ## determine the best move that the oponent (player) can play
          score, _ = self.min(alpha, beta)
          if (score > max_score):
            max_score = score
            max_x, max_y = row_index, col_index
          ## reset: remove temporary piece
          self.board[row_index][col_index] = 0
          ## alpha-beta stuff
          if BOOL_ALPHA_BETA:
            if max_score >= beta:
              return max_score, (max_x, max_y)
            if max_score > alpha:
              alpha = max_score
    return max_score, (max_x, max_y)

  def min(self, alpha, beta):
    ## check if the game has concluded
    status = self.checkGameOverStatus()
    if status is not None:
      return -1*status, (None, None)
    ## initialise to worst case
    min_score = np.inf
    min_x = None
    min_y = None
    ## loop over possible moves
    for row_index in range(3):
      for col_index in range(3):
        ## free spot on the board
        if self.board[row_index][col_index] == 0:
          ## make temporary opponent (player) move
          self.board[row_index][col_index] = 1
          ## determine the best move that the (player) can play
          score, _ = self.max(alpha, beta)
          if (score < min_score):
            min_score = score
            min_x, min_y = row_index, col_index
          ## reset: remove temporary piece
          self.board[row_index][col_index] = 0
          ## alpha-beta stuff
          if BOOL_ALPHA_BETA:
            if min_score <= alpha:
              return min_score, (min_x, min_y)
            if min_score < beta:
              beta = min_score
    return min_score, (min_x, min_y)

  def isValidMove(self, to_coord):
    x, y = to_coord
    ## check x-coord lies within board bounds
    if (x < 0) or (2 < x):
      return False
    ## check y-coord lies within board bounds
    if (y < 0) or (2 < y):
      return False
    ## check that the target cell is empty
    return self.board[y][x] == 0

  def checkGameOverStatus(self):
    ## check if a horizontal win occured
    for row_index in range(3):
      if (abs(self.board[row_index][0]) > 0) and (
          (self.board[row_index][0] == self.board[row_index][1]) and
          (self.board[row_index][0] == self.board[row_index][2])
        ): return self.board[row_index][0]
    ## check if a vertical win occured
    for col_index in range(3):
      if (abs(self.board[0][col_index]) > 0) and (
          (self.board[0][col_index] == self.board[1][col_index]) and
          (self.board[0][col_index] == self.board[2][col_index])
        ): return self.board[0][col_index]
    ## check if a top-left to bottom-right diagonal win occured
    if (abs(self.board[0][0]) > 0) and (
        (self.board[0][0] == self.board[1][1]) and
        (self.board[0][0] == self.board[2][2])
      ): return self.board[0][0]
    ## check if a top-right to bottom-left diagonal win occured
    if (abs(self.board[0][2]) > 0) and (
        (self.board[0][2] == self.board[1][1]) and
        (self.board[0][2] == self.board[2][0])
      ): return self.board[0][2]
    ## check if there are any free spots available on the board
    for row in self.board:
      for cell in row:
        if cell == 0:
          return None
    ## it is a tie
    return 0

  def printBoard(self):
    print(" ", "   ".join(["0", "1", "2"]))
    for index, row in enumerate(self.board):
      for cell in row:
        if   cell > 0: piece = "x"
        elif cell < 0: piece = "o"
        else: piece = "-"
        print(f"| {piece}", end=" ")
      print(f"| {index}")
    print(" ")

  def tests(self):
    ## test 1: detect free spots
    self.board = np.array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ])
    if self.checkGameOverStatus() is not None:
      raise Exception("Failed test 1: did not detect any available cells.")
    ## check tests work for both players
    for val in [1, -1]:
      ## test 2: diagonal win (top-left to bottom-right)
      self.board = np.array([
        [val, 0, 0],
        [0, val, 0],
        [0, 0, val]
      ])
      if not (self.checkGameOverStatus() == val):
        raise Exception(f"Failed test 2: {val}")
      ## test 3: diagonal win (top-right to bottom-left)
      self.board = np.array([
        [0, 0, val],
        [0, val, 0],
        [val, 0, 0]
      ])
      if not (self.checkGameOverStatus() == val):
        raise Exception(f"Failed test 3: {val}")
      ## test 4: row win
      for col_index in range(3):
        self.board = np.zeros((3,3))
        self.board[:, col_index] = val
        if not (self.checkGameOverStatus() == val):
          raise Exception(f"Failed test 4: {col_index} {val}")
      ## test 5: column win
      for row_index in range(3):
        self.board = np.zeros((3,3))
        self.board[row_index, :] = val
        if not (self.checkGameOverStatus() == val):
          raise Exception(f"Failed test 5: {row_index} {val}")
    ## test 6: draw
    self.board = np.array([
      [ 1,  1, -1],
      [-1,  1,  1],
      [ 1, -1, -1]
    ])
    if not (self.checkGameOverStatus() == 0):
      raise Exception("Failed test 6: did not detect draw.")
    ## success
    print("Passed all tests.")


## ###########################
## DEFINE MAIN PROGRAM
## ###########################
def main():
  game = TicTacToe()
  game.play()


## ###########################
## RUN MAIN
## ###########################
if __name__ == "__main__":
  main()


## END OF PROGRAM