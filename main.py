import random
import numpy as np
import os

from helper import MazeCell, Move
from maze import Maze
# from val_agent import ValueAgent
from util_agent import UtilityAgent
from grid_plotter import GridPlotter, plot_data_per_trial

def get_p1_maze():
    """
    Returns the fixed layout of the maze given in Assignment 1 part 1
    """

    return [
        ['G', 'W', 'G', ' ', ' ', 'G'],
        [' ', 'O', ' ', 'G', 'W', 'O'],
        [' ', ' ', 'O', ' ', 'G', ' '],
        [' ', ' ', ' ', 'O', ' ', 'G'],
        [' ', 'W', 'W', 'W', 'O', ' '],
        [' ', ' ', ' ', ' ', ' ', ' '],
    ]

def generate_maze(width, height, wall_prob=0.1, floor_prob=0.6):
    """Generate a random maze grid with walls, white, green, and brown cells."""
    grid = []

    green_orange_prob = (1 - floor_prob) / 2
    for _ in range(height):
        row = []
        for _ in range(width):
            cell_type = random.choices(
                [MazeCell.WALL.value, MazeCell.FLOOR.value, MazeCell.GREEN.value, MazeCell.ORANGE.value],
                weights=[wall_prob, floor_prob, green_orange_prob, green_orange_prob]
            )[0]
            row.append(cell_type)
        grid.append(row)
    
    return grid


def step_by_step_run(mazeGrid):
    """
    Helper function for debugging by giving the option to go step by step in the iterative process
    """
    start_pos = (3, 2)

     # Initialize maze and agent
    maze = Maze(mazeGrid)
    agent = UtilityAgent(maze=maze)
    # agent = ValueAgent(maze=maze)

    cur_state = start_pos
    agent_move = ""

    while(True):
        print("\n// Actions //")
        for i in range(len(Move) * 2 + 2):
            if(i < len(Move)):
                print(f"  {i}. See expected util for {Move(i)}")
            elif (i < 2*len(Move)):
                print(f"  {i}. {Move(i % len(Move))}")
            elif(i == 2*len(Move)):
                print(f"  {i}. Perform util update")
            elif(i == 2*len(Move) + 1):
                print(f"  {i}. Print policy table")
        
        try:
            agent_move = int(input("Check action: "))
        except:         # non int input string
            print("Exiting!")
            break

        if(agent_move >= 0 and agent_move < len(Move)):
            util = agent.get_expected_utility(cur_state, Move(agent_move))
            print(f"\nExpected utility for {Move(agent_move)}: {util}")
        elif(agent_move < 2*len(Move)):
            cur_state = agent.get_next_state(cur_state, Move(agent_move % len(Move)))
        elif(agent_move == 2*len(Move)):
            iter_type = input("Value iteration(VI) or Policy iteration (PI)? ")
            if(iter_type.lower() == 'vi'):
                agent.value_iteration()     # default min_step is 1 so it will just perform 1 run of the update
            else:
                agent.policy_iteration()
            agent.print_u_table()
        else:
            agent.print_policy()

def run_vi(mazeGrid):
     # Initialize maze and agent
    maze = Maze(mazeGrid)
    # agent = ValueAgent(maze=maze)
    agent = UtilityAgent(maze=maze)

    max_steps = 1000  # Maximum steps per episode

    utilities, policy, _ = agent.value_iteration(max_steps)

    agent.print_u_table()
    agent.print_policy()

    plotter = GridPlotter(utilities=utilities, policy=policy, save_path="plots/PartOne")
    while(True):
        try:
            print("=== Plot options ===\n  1. Plot optimal policies\n  2. Plot utilities\n  3. Plot utility estimates\n  4. Plot utility estimates by row")
            
            choice = int(input("Plot action: "))
            if(choice == 1):
                plotter.plot_optimal_policy(maze=maze, save_filename="VI_policy", show_plot=False)
            elif(choice == 2):
                plotter.plot_utility_graph(maze=maze, save_filename="VI_utility", show_plot=False)
            elif(choice == 3):
                plotter.plot_utility_estimates(maze=maze, save_filename="VI_utility_estimates", show_plot=False)
            elif(choice == 4):
                plotter.plot_utility_estimates_separate(maze=maze, save_filename="VI_utility_estimates_by_row", show_plot=False)
            else:
                break
        except:         # non int input string
            print("Exiting!")
            break

def run_pi(mazeGrid):
     # Initialize maze and agent
    maze = Maze(mazeGrid)
    # agent = ValueAgent(maze=maze)
    agent = UtilityAgent(maze=maze)

    max_steps = 1000  # Maximum steps per episode

    utilities, policy, _ = agent.policy_iteration(max_steps)
    agent.print_u_table()
    agent.print_policy()

    plotter = GridPlotter(utilities=utilities, policy=policy, save_path="plots/PartOne")

    while(True):
        try:
            print("=== Plot options ===\n  1. Plot optimal policies\n  2. Plot utilities\n  3. Plot utility estimates\n  4. Plot utility estimates by row")
            
            choice = int(input("Plot action: "))
            if(choice == 1):
                plotter.plot_optimal_policy(maze=maze, save_filename="PI_policy", show_plot=False)
            elif(choice == 2):
                plotter.plot_utility_graph(maze=maze, save_filename="PI_utility", show_plot=False)
            elif(choice == 3):
                plotter.plot_utility_estimates(maze=maze, save_filename="PI_utility_estimates", show_plot=False)
            elif(choice == 4):
                plotter.plot_utility_estimates_separate(maze=maze, save_filename="PI_utility_estimates_by_row", show_plot=False)
            else:
                break
        except:         # non int input string
            print("Exiting!")
            break

def print_grid(grid):
    for row in grid:
        for item in row:
            print(f" {item} ", end = "")
        print()
    print()

def set_maze():
    dimensions = [(8, 8), (10, 10), (12, 12), (14, 14)]

    while(True):
        try:
            print("=== Maze ===")
            print("1. Part one maze")
            for i in range(len(dimensions)):
                print(f"{i + 2}. {dimensions[i][0]} x {dimensions[i][1]}")
            
            choice = int(input("Plot action: "))
            if(choice == 1):
                return get_p1_maze()
            else:
                if(choice > 1 and choice <= len(dimensions) + 1):
                    maze = generate_maze(dimensions[choice - 2][0], dimensions[choice - 2][1], wall_prob=0.2)

                    print_grid(maze)

                    return maze
        except:         # non int input string
            print("Exiting!")
            break

def check_others():
    trials_per_dim = 5
    max_steps = 10000

    dimensions = [(8, 8), (10, 10), (11, 13), (12, 12), (14, 14), (16, 16), (18, 18), (25, 25), (50, 50), (100, 100)]

    pi_iterations_per_dim = []
    vi_iterations_per_dim = []

    pi_exectime_per_dim = []
    vi_exectime_per_dim = []


    folder_path = 'plots/PartTwo'
    for dim in dimensions:
        dim_string = f"{dim[0]}x{dim[1]}"
        os.makedirs(f"{folder_path}/{dim_string}", exist_ok=True)

        print(f"Checking {dim_string}!")
        pi_iterations = []
        vi_iterations = []

        pi_exectime = []
        vi_exectime = []

        for i in range(trials_per_dim):
            print(f"{dim_string}, Trial {i}")
            maze_grid = generate_maze(dim[0], dim[1], wall_prob=0.2)
            maze = Maze(maze_grid)

            # Value iteration
            vi_agent = UtilityAgent(maze=maze)
            vi_utilities, vi_policy, vi_time = vi_agent.value_iteration(max_steps=max_steps)
            vi_iterations.append(len(vi_utilities))
            vi_exectime.append(vi_time)

            vi_plotter = GridPlotter(utilities=vi_utilities, policy=vi_policy, save_path=f"plots/PartTwo/{dim_string}")
            vi_plotter.plot_optimal_policy(maze, save_filename=f"{dim_string}_VI_policy_{i}", show_plot=False)
            vi_plotter.plot_utility_graph(maze, save_filename=f"{dim_string}_VI_utility_{i}", show_plot=False)

            # Policy iteration
            pi_agent = UtilityAgent(maze=maze)
            pi_utilities, pi_policy, pi_time = pi_agent.policy_iteration(max_steps=max_steps)
            pi_iterations.append(len(pi_utilities))
            pi_exectime.append(pi_time)

            pi_plotter = GridPlotter(utilities=pi_utilities, policy=pi_policy, save_path=f"plots/PartTwo/{dim_string}")

            pi_plotter.plot_optimal_policy(maze, save_filename=f"{dim_string}_PI_policy_{i}", show_plot=False)
            pi_plotter.plot_utility_graph(maze, save_filename=f"{dim_string}_PI_utility_{i}", show_plot=False)

        pi_iterations_per_dim.append(pi_iterations)
        vi_iterations_per_dim.append(vi_iterations)

        pi_exectime_per_dim.append(pi_exectime)
        vi_exectime_per_dim.append(vi_exectime)

    # print(pi_iterations_per_dim)
    # print(vi_iterations_per_dim)
    for i in range(len(dimensions)):
        dim_string = f"{dimensions[i][0]}x{dimensions[i][1]}"
        plot_data_per_trial(vi_data=vi_iterations_per_dim[i], pi_data=pi_iterations_per_dim[i], title=f"Iterations for Value & Policy iteration in {dim_string}", save_filename=f"plots/PartTwo/{dim_string}/{dim_string}_iterations", show_plot=False)
        
        plot_data_per_trial(vi_data=vi_exectime_per_dim[i], pi_data=pi_exectime_per_dim[i], y_label="Time taken", title=f"Exec time for Value & Policy iteration in {dim_string}", save_filename=f"plots/PartTwo/{dim_string}/{dim_string}_exec_time", show_plot=False)



def custom():
    pass

def main():
    # Initialize maze and agent
    mazeGrid = get_p1_maze()

    while(True):
        try:
            print("1. Step by step (DEBUG)")
            print("2. Value iteration")
            print("3. Policy iteration")
            print("4. Set maze")
            print("5. Check other mazes")
            choice = int(input("Choice: "))

            if(choice == 1):
                step_by_step_run(mazeGrid)
            elif(choice == 2):
                run_vi(mazeGrid)
            elif(choice == 3):
                run_pi(mazeGrid)
            elif(choice == 4):
                mazeGrid = set_maze()
            elif(choice == 5):
                check_others()
            else:
                custom()

        except:
            print("Invalid option! Exiting...")
            break


if __name__ == "__main__":
    main()