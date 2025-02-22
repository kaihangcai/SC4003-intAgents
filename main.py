import random
import numpy as np
from helper import MazeCell, Move
from maze import Maze
from agent import MazeAgent

def get_p1_maze():
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
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)
    # start_pos = (2, 3)

     # Initialize maze and agent
    maze = Maze(mazeGrid, start_pos)
    agent = MazeAgent(maze=maze)

    cur_state = start_pos
    agent_move = ""

    while(True):
        print("\n// Actions //")
        for i in range(len(Move) * 2 + 1):
            if(i < len(Move)):
                print(f"  {i}. See expected util for {Move(i)}")
            elif (i < 2*len(Move)):
                print(f"  {i}. {Move(i % len(Move))}")
            else:
                print(f"  {i}. Perform util update")
        
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
        else:
            agent.update_utilities(steps=1)
            agent.print_u_table()

def part_one():
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)
    # start_pos = (2, 3)

     # Initialize maze and agent
    maze = Maze(mazeGrid, start_pos)
    agent = MazeAgent(maze=maze)

    max_steps = 100  # Maximum steps per episode

    agent.update_utilities(steps=max_steps)

    agent.print_u_table()
    agent.print_optimal_actions()

def main():
    try:
        print("1. Step by step")
        print("2. Value iteration")
        # print("3. Policy iteration")
        choice = int(input("Choice: "))

        if(choice == 1):
            step_by_step_run()
        elif(choice == 2):
            part_one()

    except:
        print("Invalid option! Exiting...")


if __name__ == "__main__":
    main()