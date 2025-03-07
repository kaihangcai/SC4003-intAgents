import numpy as np
import time

from helper import Move

class UtilityAgent:
    def __init__(self, maze, discount_factor=0.99, threshold=0.0001):
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
        
        self.u_table = np.zeros((maze.height, maze.width))  # stores utility values
        self.u_prime_table = np.zeros((maze.height, maze.width))    # stores updated utility values, then updates u_table at the end of a loop

        self.init_policy()
        self.init_u_prime_table()

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

    def init_u_prime_table(self):
        """
        Initializes utility values of reward/punish states based on reward values
          - Basically the u_table is initialized to show the R(s) values for each state
        """
        for rowIdx in range(self.maze.height):
            for colIdx in range(self.maze.width):
                state = (rowIdx, colIdx)
                if(self.maze.is_wall(state)):
                    self.u_prime_table[state] = 0
                else:
                    self.u_prime_table[state] = self.maze.get_reward(state)

    def calculate_policy(self):
        """
        Calculates the new optimal policy based on the calculated utilities 
        """
        for rowIdx in range(self.maze.height):
            for colIdx in range(self.maze.width):
                state = (rowIdx, colIdx)

                # no calc needed for these cells
                if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                    continue

                self.policy[rowIdx][colIdx] = Move(np.argmax([self.get_expected_utility(state, Move(i)) for i in range(len(Move))]))
                    


    def policy_iteration(self, max_steps=1):
        """
        Performs Policy Iteration to update u_table and policy accordingly
          - Policy is updated on each loop (if necessary) and the loop terminates when there are no updates left to make

        Params:
            max_steps: int, controls the maximum number of policy iterations

        Returns:
            utilities: ndarray of utility values calculated over each iteration (last entry would just be the final utility values)
            policy: the resulting optimal policies for each grid cell
            exec_time: time taken to execute the iteration function
        """
        utilities = []
        iteration = 0
        start_time = time.time()
        while(True):
            utilities.append(np.copy(self.u_table))
            self.u_table = self.u_prime_table.copy()    # assign u_table as a copy of u_prime_table

            # Policy evaluation (using cur policy, eval utilities)
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)

                    # no calc needed for these cells
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    action = self.policy[rowIdx][colIdx]
                    self.u_prime_table[state] = self.maze.get_reward(state) + self.discount_factor * self.get_expected_utility(state, action)

            # Policy improvement
            policy_stable = True
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    cur_move: Move = self.policy[rowIdx][colIdx]
                    best_move = Move(np.argmax([self.get_expected_utility(state, Move(i)) for i in range(len(Move))]))

                    if cur_move != best_move:     # a new move has been found to be more optimal
                        self.policy[rowIdx][colIdx] = best_move
                        policy_stable = False

            iteration += 1

            if(policy_stable):
                print(f"Policy Iteration converged after {iteration} loops!")
                break

            if(iteration == max_steps):
                print(f"Policy Iteration did not converge! Terminating after {iteration} loops!")
                break
        
        utilities = np.stack(utilities, axis=0)
        exec_time = time.time() - start_time    # time taken for execution

        return utilities, self.policy, exec_time

    def value_iteration(self, max_steps=1):
        """
        Performs Value Iteration to update u_table and policy accordingly
          - Policy is updated AFTER the VI step when convergence has been attained
        
        Params:
            max_steps: int, controls the maximum number of value iterations

        Returns:
            utilities: ndarray of utility values calculated over each iteration (last entry would just be the final utility values)
            policy: the resulting optimal policies for each grid cell
            exec_time: time taken to execute the iteration function
        """
        utilities = []
        iteration = 0
        start_time = time.time()
        while(True):
            delta = 0   # to track the max difference in updated values
            utilities.append(np.copy(self.u_prime_table))
            self.u_table = self.u_prime_table.copy()    # assign u_table as a copy of u_prime_table

            # for each state s in S,
            for rowIdx in range(self.maze.height):
                for colIdx in range(self.maze.width):
                    state = (rowIdx, colIdx)

                    # no calc needed for these cells
                    if(self.maze.is_wall(state) or self.maze.is_reward(state) or self.maze.is_punishment(state)):
                        continue

                    # update U' table with new utility value
                    self.u_prime_table[state] = self.get_max_expected_utility(state)
                    delta = max(delta, abs(self.u_prime_table[state] - self.u_table[state]))

            iteration += 1

            if(delta < self.threshold * (1 - self.discount_factor) / self.discount_factor):
                print(f"Value iteration converged after {iteration} loops!")
                break

            if(iteration == max_steps):
                print(f"Value Iteration did not converge! Terminating after {iteration} loops!")
                break

        utilities = np.stack(utilities, axis=0)

        # calculate actual policy using new utilities
        self.calculate_policy()
        exec_time = time.time() - start_time

        return utilities, self.policy, exec_time

    def get_max_expected_utility(self, state):
        """
            Finds max. expected utility and the corresponding optimal move for a given state (goes through all possible actions)

            Params:
                state: Tuple[int, int] indicating the position (a.k.a state)

            Returns:
                max_exp_util: Max. expected utility (float)
        """
        max_exp_util = max([self.get_expected_utility(state, Move(i)) for i in range(len(Move))])
        
        # R(s) + y * max(EU(s') for all s')
        return self.maze.get_reward(state) + self.discount_factor * max_exp_util
            
    def get_expected_utility(self, state, action):
        """
        Finds expected utility for a given state + action

        Params:
            state: Tuple[int, int] representing CURRENT state
            action: Move() representing the policy(state) value
        """
        probabilities = [0.8, 0.1, 0.1]
        util = 0

        actions = [action]    # stores all the possible action(s) given the initial action choice
        actions.extend(self.get_lateral_moves(action))  # get the sideways movement options

        # calculate the utility for this particular chosen action
        for j in range(len(actions)):
            next_state = self.get_next_state(state, actions[j])  
            util += probabilities[j] * self.u_table[next_state]

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
        self.calculate_policy()

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