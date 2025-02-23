import random
import numpy as np
from helper import MazeCell, Move
from maze import Maze
from agent import MazeAgent

def get_p1_maze():
    """
    Returns the fixed layout of the maze given in Assignment 1 part 1
    """

    return [
        ['G', 'W', 'G', ' ', ' ', 'G'],
        [' ', 'B', ' ', 'G', 'W', 'B'],
        [' ', ' ', 'B', ' ', 'G', ' '],
        [' ', ' ', ' ', 'B', ' ', 'G'],
        [' ', 'W', 'W', 'W', 'B', ' '],
        [' ', ' ', ' ', ' ', ' ', ' '],
    ]

def generate_maze(width, height, start_pos, wall_prob=0.2):
    """Generate a random maze grid with walls, white, green, and brown cells."""
    grid = []
    for _ in range(height):
        row = []
        for _ in range(width):
            cell_type = random.choices(
                [MazeCell.WALL.value, MazeCell.WHITE.value, MazeCell.GREEN.value, MazeCell.BROWN.value],
                weights=[wall_prob, 0.6, 0.1, 0.1]
            )[0]
            row.append(cell_type)
        grid.append(row)
    
    # start position should be white
    grid[start_pos] = MazeCell.WHITE.value
    return grid


def step_by_step_run():
    """
    Helper function for debugging by giving the option to go step by step in the iterative process
    """
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)

     # Initialize maze and agent
    maze = Maze(mazeGrid)
    agent = MazeAgent(maze=maze)

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
            iter_type = input("Value iteration(VI) or Policy iteration (PI)?")
            if(iter_type.lower() == 'vi'):
                agent.value_iteration()     # default min_step is 1 so it will just perform 1 run of the update
            else:
                agent.policy_iteration()
            agent.print_u_table()
        else:
            agent.print_policy()

def part_one_vi():
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)

     # Initialize maze and agent
    maze = Maze(mazeGrid)
    agent = MazeAgent(maze=maze)

    max_steps = 100  # Maximum steps per episode

    agent.value_iteration(max_steps)

    agent.print_u_table()
    agent.print_policy()

def part_one_pi():
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)

     # Initialize maze and agent
    maze = Maze(mazeGrid)
    agent = MazeAgent(maze=maze)

    max_steps = 100  # Maximum steps per episode

    agent.policy_iteration(max_steps)

    agent.print_u_table()
    agent.print_policy()

def custom():
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)

     # Initialize maze and agent
    maze = Maze(mazeGrid)
    agent = MazeAgent(maze=maze)

    print("Print policy!")
    agent.value_iteration(1)

def main():
    try:
        print("1. Step by step (DEBUG)")
        print("2. Value iteration")
        print("3. Policy iteration")
        choice = int(input("Choice: "))

        if(choice == 1):
            step_by_step_run()
        elif(choice == 2):
            part_one_vi()
        elif(choice == 3):
            part_one_pi()
        else:
            custom()

    except:
        print("Invalid option! Exiting...")


if __name__ == "__main__":
    main()