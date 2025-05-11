import helper


class Maze:
    maze_matrix = []
    begin_pos = ()
    end_pos = ()
    agent_pos = ()

    def __init__(self, matrix):
        self.maze_matrix = matrix
        self.begin_pos = self.find_begin(self.maze_matrix)
        self.end_pos = self.find_end(self.maze_matrix)
        self.agent_pos = self.begin_pos

    def get_matrix(self):
        return self.maze_matrix

    def find_begin(self):
        for i in len(self.maze_matrix):
            for j in len(self.maze_matrix[0]):
                if self.maze_matrix[i][j] == "b":
                    return (i, j)

    def get_begin(self):
        if len(self.begin_pos) == 0:
            return self.find_begin()
        return self.begin_pos

    def get_end(self):
        if len(self.end_pos) == 0:
            return self.find_end()
        return self.end_pos

    def get_agent_pos(self):
        return self.agent_pos

    def find_end(self):
        for i in len(self.maze_matrix):
            for j in len(self.maze_matrix[0]):
                if self.maze_matrix[i][j] == "e":
                    return (i, j)

    def is_begin(self):
        pos = self.agent_pos
        return self.maze_matrix[pos[0]][pos[1]] == "b"

    def is_end(self):
        pos = self.agent_pos
        return self.maze_matrix[pos[0]][pos[1]] == "e"

    def step(self, action):
        reward = 0
        # TODO implement step
        return reward
