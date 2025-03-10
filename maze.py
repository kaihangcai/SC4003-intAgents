from helper import MazeCell

class Maze:
    def __init__(self, grid):
        """
        Initialize the maze with a grid representation.
        
        Params:
            grid: 2d array of distinct strings representing the grid environment
        """
        self.grid = grid  # 2D list representing the maze layout
        self.height = len(grid)
        self.width = len(grid[0])
    
    def is_wall(self, position):
        """Check if a given state is a wall."""
        rowIdx, colIdx = position
        return self.grid[rowIdx][colIdx] == MazeCell.WALL.value
    
    def is_out_of_bounds(self, position):
        """Check if given state is out of bounds from the maze environment"""
        rowIdx, colIdx = position
        is_oob = not (rowIdx >= 0 and rowIdx < self.height and colIdx >= 0 and colIdx < self.width)
        return is_oob
    
    def is_reward(self, position):
        rowIdx, colIdx = position
        return self.grid[rowIdx][colIdx] == MazeCell.GREEN.value
    
    def is_punishment(self, position):
        rowIdx, colIdx = position
        return self.grid[rowIdx][colIdx] == MazeCell.ORANGE.value

    def get_reward(self, position):
        """Get the reward for a given state."""
        rowIdx, colIdx = position
        cell = self.grid[rowIdx][colIdx]
        if cell == MazeCell.GREEN.value:  # Green square
            return 1
        elif cell == MazeCell.ORANGE.value:  # Orange square
            return -1
        else:  # White square
            return -0.05
        
    def print_grid(self, current_pos):
        """Prints the current grid view"""
        for rowIdx in range(self.height):
            column_headers = [idx for idx in range(len(self.grid[rowIdx]))]
            if(rowIdx == 0): # print col index
                print("   ", end="")
                for header in column_headers:
                    print(f" {header} ", end="")
                print()

            for colIdx in range(self.width):
                if(colIdx == 0):
                    print(f"{rowIdx}  ", end="")     # print row index

                if(rowIdx, colIdx) == current_pos:
                    print(" X ", end="")
                else:
                    print(f" {self.grid[rowIdx][colIdx]} ", end="")
            print()
        print(f"Current state: {current_pos}")

    def print_grid_rewards(self, current_pos):
        """Prints the current grid but showing the reward values instead"""
        for rowIdx in range(self.height):
            column_headers = [idx for idx in range(len(self.grid[rowIdx]))]
            if(rowIdx == 0): # print col index
                print("   ", end="")
                for header in column_headers:
                    print(f" {header} ", end="")
                print()

            for colIdx in range(self.width):
                if(colIdx == 0):
                    print(f"{rowIdx}  ", end="")     # print row index

                if(rowIdx, colIdx) == current_pos:
                    print(" X ", end="")
                else:
                    print(f" {self.get_reward((rowIdx, colIdx))} ", end="")
            print()
        print(f"Current state: {current_pos}")