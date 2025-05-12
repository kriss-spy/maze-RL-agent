import helper


class Maze:
    maze_matrix = []
    begin_pos = ()
    end_pos = ()
    agent_pos = ()
    end_reward = 0
    visited_set = set()
    visited_stack = []

    def __init__(self, matrix, settings):
        self.maze_matrix = matrix
        self.begin_pos = self.find_begin()
        self.end_pos = self.find_end()
        self.agent_pos = self.begin_pos
        self.end_reward = settings["end_reward"]
        self.visited_set = set()
        self.visited_stack = []
        self.update_visited((0, 0))

    def get_matrix(self):
        return self.maze_matrix

    def find_begin(self):
        for i in range(len(self.maze_matrix)):
            for j in range(len(self.maze_matrix[0])):
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

    def at(self, pos):
        return self.maze_matrix[pos[0]][pos[1]]

    def find_end(self):
        for i in range(len(self.maze_matrix)):
            for j in range(len(self.maze_matrix[0])):
                if self.maze_matrix[i][j] == "e":
                    return (i, j)

    def is_begin(self):
        return self.at(self.agent_pos) == "b"

    def is_end(self):
        return self.at(self.agent_pos) == "e"

    def step(self, action):
        reward = 0
        self.agent_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
        foo = self.at(self.agent_pos)
        reward = self.end_reward if foo == "e" else foo
        return reward

    def state(self):
        return self  # assume that env (containing maze_matrix and agent_pos) is state

    def valid(self, action):
        new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
        if not (
            (0 <= new_pos[0] < len(self.maze_matrix))
            and (0 <= new_pos[1] < len(self.maze_matrix[0]))
        ):
            return False
        if new_pos in self.visited_set:
            return False
        if self.at(new_pos) == "b" or self.at(new_pos) == "e":
            return True
        if self.at(new_pos) < 0:
            return False
        return True

    def update_visited(self, action):
        new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
        self.visited_set.add(new_pos)
        self.visited_stack.append(new_pos)
