import numpy as np
import random
import sys

from helper import Move

class MazeAgent:
    def __init__(self, maze, discount_factor=0.99, epsilion=0.001):
        self.maze = maze
        self.discount_factor = discount_factor  # Discount factor (gamma)

        self.epsilion = epsilion
        
        self.u_table = np.zeros((maze.height, maze.width))


    # modify to use value iteration / policy iteration
    def update_utilities(self, steps=10):
        has_converged = True

        for i in range(steps):
            # if(i % 10 == 0):
            # print(f"Step: {i}")
            for y in range(self.maze.height):
                for x in range(self.maze.width):
                    state = (x, y)
                    # print(f"{state}: {self.u_table[state]}")

                    if(self.maze.is_wall(state)):
                        continue

                    max_util, _ = self.get_max_expected_utility(state)
                    # print(f"Max util: {max_util}")
                    R_s = self.maze.get_reward(state)
                    # print(f"Cell ({state}): {R_s}")
                    new_util = R_s + self.discount_factor * max_util # update util table

                    if(has_converged):  # update convergence check
                        has_converged = abs(new_util - self.u_table[state]) < self.epsilion

                    self.u_table[state] = new_util

            if(has_converged):  # terminate early if convergence achieved
                print("Early termination!")
                break


    def get_max_expected_utility(self, state):
        """
            Finds max. expected utility for a given state (goes through all possible actions)
            Returns:
                max_util: Max. possible utility (float probably)
                best_move: Move.[UP/DOWN/LEFT/RIGHT]
        """
        best_move = None
        max_util = sys.float_info.min

        for i in range(len(Move)):
            util = self.get_expected_utility(state, Move(i))

            if(util > max_util):
                max_util = util
                best_move = Move(i)

        return max_util, best_move
            
    # finds the expected utility given a state + action
    def get_expected_utility(self, state, action):
        probabilities = [0.8, 0.1, 0.1]
        util = 0

        actions = [action]    # stores all the possible action(s) given the initial action choice
        actions.extend(self.get_lateral_moves(action))  # get the sideways movement options

        # calculate the utility for this particular chosen action
        for j in range(len(actions)):
            util += probabilities[j] * self.u_table[self.get_next_state(state, actions[j])]

        return util

    def get_next_state(self, cur_state, action):
        """Get the next state based on the action."""
        x, y = cur_state

        if action == Move.UP:
            next_state = (x, y - 1)
        elif action == Move.RIGHT:
            next_state = (x + 1, y)
        elif action == Move.DOWN:
            next_state = (x, y + 1)
        elif action == Move.LEFT:
            next_state = (x - 1, y)
        else:
            next_state = cur_state  # Invalid action
        
        # if state val is out of bounds of the env, or is a wall - return the original position (no state transition)
        if (self.maze.is_out_of_bounds(next_state) or self.maze.is_wall(next_state)):
            return cur_state
        return next_state

    def get_lateral_moves(self, action):
        """Get possible lateral moves (right angles to the intended move)."""
        lateral_actions = {
            Move.UP: [Move.RIGHT, Move.LEFT],
            Move.DOWN: [Move.RIGHT, Move.LEFT],
            Move.LEFT: [Move.UP, Move.DOWN],
            Move.RIGHT: [Move.UP, Move.DOWN]
        }

        return lateral_actions[action]
    
    # helper function to show the utility table stored so far
    def print_u_table(self):
        for i in range(len(self.u_table)):
            row = self.u_table[i]
            column_headers = [idx for idx in range(len(row))]
            if(i == 0): # print col index
                print("   ", end="")
                for header in column_headers:
                    print(f"  {header}  ", end="")
                print()
            
            
            for j in range(len(row)):
                if(j == 0):
                    print(f"{i}  ", end="")     # print row index
                print(f" {row[j]} ", end="")
            print()