import helper


class Maze:
    maze_matrix = []

    def __init__(self, matrix):
        self.maze_matrix = matrix

    def get_matrix(self):
        return self.maze_matrix
