import matplotlib.pyplot as plt
import numpy as np

from helper import MazeCell, Move

class GridPlotter:
    def __init__(self, utilities, policy):
        """
        Initializes the plotter with the agent object
        """
        self.utilities = utilities
        self.policy = policy

    def plot_utility_graph(self, maze):
        """
        Plots the grid world with the calculated utilities displayed on the white cells
        """

        cell_text = self.get_utility_dict(maze)
        cell_colors = self.get_color_dict(maze)

        self.plot_gridworld(rows=maze.height, cols=maze.width, cell_colors=cell_colors, cell_text=cell_text)

    def plot_optimal_policy(self, maze):
        """
        Plots the grid world with the optimal policies displayed on the white cells
        """

        cell_text = self.get_utility_dict(maze)
        cell_policy = self.get_policy_dict(maze)
        cell_colors = self.get_color_dict(maze)

        self.plot_gridworld(rows=maze.height, cols=maze.width, cell_colors=cell_colors, cell_text=cell_text, cell_actions=cell_policy)

    def get_policy_dict(self, maze):
        """
        Generates a Dictionary for the optimal policy to be used on each floor cell

        Params:
            policy: 2D policy array with the optimal actions for each floor cell
            maze: Custom maze object with maze specific information + cell checking logic

        Returns:
            cell_actions: Dictionary mapping states (the (y, x) positions) to their optimal action
        """
        def get_move_string(move):
            if(move == Move.UP):
                return "up"
            elif(move == Move.DOWN):
                return "down"
            elif(move == Move.LEFT):
                return "left"
            elif(move == Move.RIGHT):
                return "right"
            else:
                return ""

        cell_actions = {}
        for rowIdx in range(maze.height):
            for colIdx in range(maze.width):
                state = (rowIdx, colIdx)

                cell_actions[state] = get_move_string(self.policy[rowIdx][colIdx])

        return cell_actions

    def get_color_dict(self, maze):
        """
        Generates a Dictionary for the colors to be used for each cell

        Params:
            maze: Custom maze object with maze specific information + cell checking logic

        Returns:
            cell_colors: Dictionary mapping states (the (y, x) positions) to their colors
        """
        cell_colors = {}
        for rowIdx in range(maze.height):
            for colIdx in range(maze.width):
                state = (rowIdx, colIdx)

                grid_entry = maze.grid[rowIdx][colIdx]

                if grid_entry == MazeCell.ORANGE.value:
                    cell_colors[state] = "orange"
                elif grid_entry == MazeCell.GREEN.value:
                    cell_colors[state] = "green"
                elif grid_entry == MazeCell.WALL.value:
                    cell_colors[state] = "grey"
                else:
                    cell_colors[state] = "white"

        return cell_colors

    def get_utility_dict(self, maze):
        """
        Generates a Dictionary for the utility values to be displayed on each cell
          - Should also show the +1 / -1 values for reward / punishment cells

        Params:
            utility_table: 2D array u_table containing the utility values for each cell
            maze: Custom maze object with maze specific information + cell checking logic

        Returns:
            cell_text: Dictionary mapping states (the (y, x) positions) to their text
        """

        cell_text = {}
        utility_table = self.utilities[-1]  # get last entry of utilities
        for rowIdx in range(maze.height):
            for colIdx in range(maze.width):
                state = (rowIdx, colIdx)

                if(utility_table[state] == 0):
                    cell_text[state] = "Wall"
                elif(utility_table[state] == -1):
                    cell_text[state] = '-1'
                elif(utility_table[state] == 1):
                    cell_text[state] = '+1'
                else:
                    cell_text[state] = round(utility_table[state], 2)

        return cell_text

    def plot_gridworld(self, rows, cols, cell_colors=None, cell_text=None, cell_actions=None):
        """
        Plots the Gridworld environment as specified by grid
        
        Params:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            cell_colors (dict): Dictionary mapping (row, col) tuples to color strings (e.g., {(1,1): 'green'}).
            cell_text (dict): Dictionary mapping (row, col) tuples to text strings (e.g., {(1,1): '+1'}).
            cell_actions (dict): Dictionary mapping (row, col) tuples to text strings (e.g., {(1,1): 'down'}).
        """
        
        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)

        arrow_dir = {
            'up': (0, 0.3),
            'down': (0, -0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0)
        }
        
        # Draw grid cells
        for row in range(rows):
            for col in range(cols):
                state = (row, col)

                color = cell_colors.get(state, 'white')  # Default to white if not specified
                rect = plt.Rectangle((col, rows - row - 1), 1, 1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)

                rendered_arrow = False
                # IF action is given, render the corresponding arrow over rendering the utility text
                if cell_actions and state in cell_actions:
                    direction = cell_actions[state]
                    if direction in arrow_dir:
                        dx, dy = arrow_dir[direction]
                        ax.arrow(col + 0.5 - dx/2, rows - row - 0.5 - dy/2, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')
                        rendered_arrow = True

                if cell_text and state in cell_text and not rendered_arrow:
                    ax.text(col + 0.5, rows - row - 0.5, cell_text[state], 
                            ha='center', va='center', fontsize=12, fontweight='bold')
                    
                

        # Remove axis labels and ticks
        ax.set_xticks(np.arange(cols + 1))
        ax.set_yticks(np.arange(rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='black')
        ax.set_frame_on(False)
        
        plt.show()

    def plot_utility_estimates_separate(self, maze):
        """
        Plots the graph of the utility for each cell over each iteration
        """
        iterations = len(self.utilities)

        Y_OFFSET_INCREMENT = 0.05
        plotted_points = []     # store the values of the already plotted points

        fig, ax = plt.subplots(3, 2, figsize=(20,15))

        # get individual cell utilities over iterations
        for rowIdx in range(maze.height):       
            for colIdx in range(maze.width):
                utility_over_iter = self.utilities[:, rowIdx, colIdx]

                y = rowIdx // 2
                x = rowIdx % 2

                ax[y][x].plot(range(iterations), utility_over_iter)

                # Attaching position label at the end of each line plot
                y_offset = 0  # y offset to re-position text
                x_offset = 0.05

                last_x = iterations - 1
                last_y = utility_over_iter[-1]

                for _, prev_y in plotted_points:
                    if abs(prev_y - last_y) < Y_OFFSET_INCREMENT:  # If values are close, increase offset
                        y_offset += Y_OFFSET_INCREMENT
                        x_offset *= -1

                last_y += y_offset  # Adjust label position
                last_x += x_offset
                plotted_points.append((last_x, last_y))  # Store adjusted label position
                
                ax[y][x].text(last_x, last_y, f"({rowIdx},{colIdx})", fontsize=7, verticalalignment='center')
            ax[y][x].set_xlabel("Number of iterations")
            ax[y][x].set_ylabel("Utility estimates")

        plt.show()

    def plot_utility_estimates(self, maze):
        """
        Plots the graph of the utility for each cell over each iteration
        """
        iterations = len(self.utilities)

        Y_OFFSET_INCREMENT = 0.03
        plotted_points = []     # store the values of the already plotted points

        # get individual cell utilities over iterations
        for rowIdx in range(maze.height):       
            for colIdx in range(maze.width):
                utility_over_iter = self.utilities[:, rowIdx, colIdx]

                plt.plot(range(iterations), utility_over_iter, label=f"({rowIdx},{colIdx})", alpha=0.7)      

                # Attaching position label at the end of each line plot
                y_offset = 0  # y offset to re-position text
                x_offset = 0.05

                last_x = iterations - 1
                last_y = utility_over_iter[-1]

                for _, prev_y in plotted_points:
                    if abs(prev_y - last_y) < Y_OFFSET_INCREMENT:  # If values are close, increase offset
                        y_offset += Y_OFFSET_INCREMENT
                        x_offset *= -1

                last_y += y_offset  # Adjust label position
                last_x += x_offset
                plotted_points.append((last_x, last_y))  # Store adjusted label position

                angle = 15 if x_offset > 0 else -15
                
                plt.text(last_x, last_y, f"({rowIdx},{colIdx})", fontsize=7, verticalalignment='center', rotation=angle)

        plt.xlabel("Number of iterations")
        plt.ylabel("Utility estimates")
        plt.title("Utility estimates over iterations")

        plt.yticks(np.arange(-1, 1.1, 0.1))
        plt.xticks(range(iterations))

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))  # Keeps legend outside the main plot
        plt.grid(True)
        plt.show()
