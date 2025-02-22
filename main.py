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

def generate_maze(width, height, wall_prob=0.2):
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
    
    # Ensure the start and end positions are not walls
    grid[0][0] = MazeCell.WHITE.value
    grid[height - 1][width - 1] = MazeCell.GREEN.value
    return grid


def step_by_step_run():
    mazeGrid = get_p1_maze()
    start_pos = (3, 2)
    # start_pos = (2, 3)

     # Initialize maze and agent
    maze = Maze(mazeGrid, start_pos)
    agent = MazeAgent(maze=maze)

    maze.print_grid(start_pos)
    maze.print_grid_rewards(start_pos)

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
    maze = get_p1_maze()
    start_pos = (2, 3)

     # Initialize maze and agent
    maze = Maze(maze, start_pos)
    agent = MazeAgent(state_size=maze.height * maze.width, action_size=len(Move))

    num_episodes = 1  # Number of episodes to train
    max_steps = 100  # Maximum steps per episode

    for episode in range(num_episodes):
        cur_state = maze.start_pos
        for step in range(max_steps):
            print_grid(maze, cur_state)

            action = agent.choose_action(cur_state)  # Choose an action
            print(f"Chosen action: {Move(action)}")

            next_state, reward = agent.transition(cur_state, Move(action), maze)  # Apply transition
            agent.update_q_value(cur_state, action, reward, next_state)  # Update Q-table
            
            cur_state = next_state  # Move to the next state

            if step == max_steps - 1:  # End of episode
                break
        
        agent.decay_exploration()  # Reduce exploration over time

    # Display the learned policy
    print("\nLearned Policy:")
    for y in range(maze.height):
        for x in range(maze.width):
            state = (x, y)
            if maze.grid[y][x] == MazeCell.WALL.value:
                print(" W ", end="")
            else:
                best_action = Move(np.argmax(agent.q_table[state]))
                print(f" {best_action.name[0]} ", end="")
        print()

def main():
    step_by_step_run()
    # mazeGrid = get_p1_maze()
    # start_pos = (3, 2)

    #  # Initialize maze and agent
    # maze = Maze(mazeGrid, start_pos)
    # agent = MazeAgent(maze=maze)

    # maze.print_grid(start_pos)


if __name__ == "__main__":
    main()