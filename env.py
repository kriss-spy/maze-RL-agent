import helper


class Maze:
    maze_matrix = []
    begin_pos = ()
    end_pos = ()
    agent_pos = ()
    end_reward = 0
    step_cost = 0
    # removed visited_set and visited_stack

    def __init__(self, matrix, settings):
        self.maze_matrix = matrix
        self.begin_pos = self.find_begin()
        self.end_pos = self.find_end()
        self.agent_pos = self.begin_pos
        self.end_reward = settings["end_reward"]
        self.step_cost = settings["step_cost"]
        # removed visited_set and visited_stack initialization

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
        current_cell_value = self.at(self.agent_pos)  # Renamed foo for clarity
        # Check if current_cell_value is a string (e.g., 'e') or number
        if isinstance(current_cell_value, str) and current_cell_value == "e":
            reward = self.end_reward
        elif (
            isinstance(current_cell_value, (int, float)) and current_cell_value < 0
        ):  # Obstacle
            # Penalize for hitting an obstacle, could be a specific penalty or just step_cost
            # For now, let's assume hitting an obstacle gives a large negative reward or ends the episode.
            # The original code implies obstacles are handled by 'valid', so step might not see this.
            # However, if 'valid' allows stepping into an obstacle cell that then gives a reward, it's handled here.
            # Based on the 'valid' method, agent should not step into obstacle.
            # If it somehow does, or if maze can have negative rewards not obstacles:
            reward = self.step_cost  # Or a specific obstacle penalty
        else:  # Empty cell or cell with positive reward (not 'e')
            reward = self.step_cost
            if isinstance(current_cell_value, (int, float)) and current_cell_value > 0:
                reward += current_cell_value  # Add reward from the cell

        return reward

    def state(self):
        return self  # assume that env (containing maze_matrix and agent_pos) is state

    def valid(
        self, action
    ):  # Renamed from is_valid_move for consistency with existing calls
        new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
        if not (
            (0 <= new_pos[0] < len(self.maze_matrix))
            and (0 <= new_pos[1] < len(self.maze_matrix[0]))
        ):
            return False
        # Removed: if new_pos in self.visited_set:
        # Removed:     return False

        # Allow moving to 'b' or 'e'
        if self.at(new_pos) == "b" or self.at(new_pos) == "e":
            return True

        # Check for obstacles (numeric and less than 0)
        # Cells with rewards (positive numbers) or empty cells (0) are valid.
        if isinstance(self.at(new_pos), (int, float)) and self.at(new_pos) < 0:
            return False
        return True

    # Removed update_visited method
    # def update_visited(self, action):
    #     new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
    #     self.visited_set.add(new_pos)
    #     self.visited_stack.append(new_pos)

    def is_obstacle(self, pos):
        """Checks if the given position is an obstacle."""
        if not (
            (0 <= pos[0] < len(self.maze_matrix))
            and (0 <= pos[1] < len(self.maze_matrix[0]))
        ):
            return True  # Out of bounds is like an obstacle
        cell_value = self.at(pos)
        return isinstance(cell_value, (int, float)) and cell_value < 0

    def is_valid_move(self, current_pos, action):
        """Checks if an action is valid from a given position."""
        new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
        if not (
            (0 <= new_pos[0] < len(self.maze_matrix))
            and (0 <= new_pos[1] < len(self.maze_matrix[0]))
        ):
            return False

        cell_value = self.at(new_pos)
        if isinstance(cell_value, (int, float)) and cell_value < 0:  # Obstacle
            return False
        return True
