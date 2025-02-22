from enum import Enum

class Move(Enum):   # WASDs
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class MazeCell(Enum):
    FLOOR = ' '
    WALL = 'W'
    GREEN = 'G'     # +1
    BROWN = 'B'     # -1
