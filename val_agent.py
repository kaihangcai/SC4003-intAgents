import numpy as np
import sys

from helper import Move

class ValueAgent:
    """
    !!! NOT IN USE !!!

    Implements Value and Policy Iteration by calculating state values V(s) and the Q table Q(s, a), taught in SC3000
      - Achieves more or less the same end result as the Utility function in SC4003, but the calculation is quite different even though they do look fairly similar
    """
    def __init__(self, maze, discount_factor=0.99, threshold=0.01):
        """
        Initializes the agent to have knowledge of the maze + relevant hyperparams
        
        Params:
            maze: Custom Maze type with helper functions to describe the cells present in the given maze
            discount_factor: Gamma value to reduce the "importance" of future state utilities
            threshold: Threshold to check for convergence
        """
        self.maze = maze
        self.discount_factor = discount_factor  # Discount factor (gamma)

        self.threshold = threshold
        
        self.u_table = np.zeros((maze.height, maze.width))
        self.init_policy()

    def init_policy(self):
        """
        Initializes the policy table at the start with a "placeholder" move
        """
        self.policy = []

        # init policies
        for rowIdx in range(len(self.u_table)):
            row = self.u_table[rowIdx]
            policy_row = []
            for colIdx in range(len(row)):
                state = (rowIdx, colIdx)
                if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                    policy_row.append(None)
                else:
                    policy_row.append(Move(0))
            self.policy.append(policy_row)

    def policy_iteration(self, min_steps=1):
        """
        Performs Policy Iteration to update u_table and policy accordingly

        Params:
            min_steps: int, controls the number of policy iterations
        """
        print("Policy iteration!")

        iteration = 0
        while(True):
            # Policy evaluation
            delta = 0   # to track the max difference in updated values
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)

                    # no calc needed for these cells
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    cur_util = self.u_table[state]
                    action = self.policy[state[0]][state[1]]

                    util = self.get_expected_utility(state, action)
                    self.u_table[state] = util

            # Policy improvement
            policy_stable = True
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    _, best_move, has_updated = self.get_max_expected_utility(state)

                    if has_updated:     # a new move has been found to be more optimal
                        self.policy[rowIdx][colIdx] = best_move
                        policy_stable = False

            iteration += 1
            if(iteration == min_steps):
                if policy_stable:
                    print("Policy Iteration converged!")
                else:
                    print("Policy Iteration did not converge!")
                break

    def value_iteration(self, min_steps=1):
        """
        Performs Value Iteration to update u_table and policy accordingly

        Params:
            min_steps: int, controls the number of value iterations
        """
        print("Value iteration!")

        iteration = 0
        while(True):
            delta = 0   # to track the max difference in updated values

            # for each state s in S,
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)

                    # no calc needed for these cells
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    cur_util = self.u_table[state]

                    max_util, best_move, _ = self.get_max_expected_utility(state)
                    self.u_table[state] = max_util

                    self.policy[rowIdx][colIdx] = best_move     # update policy
                    delta = max(delta, abs(cur_util - max_util))
            iteration += 1
            if(iteration == min_steps):
                if delta <= self.threshold:
                    print("Value iteration converged!")
                else:
                    print("Value iteration did not converge!")
                break

    def get_max_expected_utility(self, state):
        """
            Finds max. expected utility and the corresponding optimal move for a given state (goes through all possible actions)

            Params:
                state: Tuple[int, int] indicating the position

            Returns:
                max_util: Max. possible utility (float)
                best_move: Move.[UP/DOWN/LEFT/RIGHT]
                new_optimal_move: boolean (whether or not a new move is selected)
        """
        cur_move = self.policy[state[0]][state[1]]      # get the existing policy
        best_move = None

        max_util = sys.float_info.min

        for i in range(len(Move)):
            util = self.get_expected_utility(state, Move(i))

            if(util > max_util):
                max_util = util
                best_move = Move(i)

        new_optimal_move = False    # tracks whether a new move has been selected or not
        if best_move != None and best_move != cur_move:
            new_optimal_move = True
        
        if best_move == None:
            best_move = cur_move

        return max_util, best_move, new_optimal_move
            
    def get_expected_utility(self, state, action):
        """
        Finds expected utility for a given state + action

        Params:
            state: Tuple[int, int] representing CURRENT state
            action: Move() representing the policy(state) value
        """

        print("Get expected utility!")
        probabilities = [0.8, 0.1, 0.1]
        util = 0

        actions = [action]    # stores all the possible action(s) given the initial action choice
        actions.extend(self.get_lateral_moves(action))  # get the sideways movement options

        # calculate the utility for this particular chosen action
        for j in range(len(actions)):
            next_state = self.get_next_state(state, actions[j])  
            util += probabilities[j] * (self.maze.get_reward(next_state) + self.discount_factor * self.u_table[next_state])

        return util
    
    def get_next_state(self, cur_state, action):
        """
        Get the next state based on the current state + action.
        Also accounts for the possibility of going out of bounds / into a wall.

        Returns:
            next_state: Tuple[int, int] for the resulting state
        """
        y, x = cur_state

        if action == Move.UP:
            next_state = (y - 1, x)
        elif action == Move.RIGHT:
            next_state = (y, x + 1)
        elif action == Move.DOWN:
            next_state = (y + 1, x)
        elif action == Move.LEFT:
            next_state = (y, x - 1)
        else:
            next_state = cur_state  # Invalid action
        
        # if state val is out of bounds of the env, or is a wall - return the original position (no state transition)
        if (self.maze.is_out_of_bounds(next_state) or self.maze.is_wall(next_state)):
            return cur_state
        return next_state

    def get_lateral_moves(self, action):
        """
        Get all possible lateral moves (right angles to the intended move).

        Returns:
            Array of the lateral moves of type Move()
        """
        lateral_actions = {
            Move.UP: [Move.RIGHT, Move.LEFT],
            Move.DOWN: [Move.RIGHT, Move.LEFT],
            Move.LEFT: [Move.UP, Move.DOWN],
            Move.RIGHT: [Move.UP, Move.DOWN]
        }

        return lateral_actions[action]
    
    def print_u_table(self):
        """
        Helper function to show the utility table stored so far
        """
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
        print()

    def print_policy(self):
        """
        Helper function to print the optimal action (policy) so far
        """
        for rowIdx in range(self.maze.height):
            column_headers = [idx for idx in range(len(self.maze.grid[rowIdx]))]
            if(rowIdx == 0): # print col index
                print("   ", end="")
                for header in column_headers:
                    print(f" {header} ", end="")
                print()

            for colIdx in range(self.maze.width):
                if(colIdx == 0):
                    print(f"{rowIdx}  ", end="")     # print row index

                print(f" {self.policy[rowIdx][colIdx]} ", end="")
            print()
        print()