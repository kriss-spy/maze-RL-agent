import helper
import random
from env import Maze


class Agent:
    action_list = []
    q_table = {}
    exploration_rate = 0
    default_value = 0
    discount_factor = 0
    learning_rate = 0
    steps = 0

    def __init__(self, settings):
        self.action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.q_table = {}
        self.exploration_rate = settings["exploration_rate"]
        self.default_value = settings["default_value"]
        self.discount_factor = settings["discount_factor"]
        self.learning_rate = settings["learning_rate"]
        self.steps = 0

    def value(self, state, action):
        agent_pos = state.get_agent_pos()
        # Use agent position as the state representation in q_table instead of Maze object
        if (agent_pos, action) in self.q_table.keys():
            return self.q_table[(agent_pos, action)]
        else:
            self.q_table[(agent_pos, action)] = self.default_value
            return self.default_value

    def act(self, state):
        agent_act = None
        rn = random.random()
        best_act = None
        max_val = float("-inf")
        valid_action_list = [act for act in self.action_list if state.valid(act)]
        if len(valid_action_list) == 0:
            while True:
                last_visited = state.visited_stack.pop()
                self.steps += 1
                # state.visited_set.remove(last_visited)
                state.agent_pos = last_visited
                valid_action_list = [
                    act for act in self.action_list if state.valid(act)
                ]
                if len(valid_action_list) != 0:
                    break

        if rn > self.exploration_rate:  # exploit
            for action in valid_action_list:
                val = self.value(state, action)
                if val > max_val:
                    best_act = action
                    max_val = val
            agent_act = best_act
        else:
            agent_act = random.choice(valid_action_list)
        return agent_act

    def reverse_act(self, action):
        return (
            0 if action[0] == 0 else -1 * action[0],
            0 if action[1] == 0 else -1 * action[1],
        )

    def inf_walk(self, maze: Maze, verbose=True):
        self.steps = 0
        discount_factor = self.discount_factor
        rewards = []
        while not maze.is_end():
            if verbose:
                print(
                    f"agent pos: {maze.get_agent_pos()} [{maze.at(maze.get_agent_pos())}]"
                )
                if (
                    maze.at(maze.get_agent_pos()) not in ["b", "e"]
                    and maze.at(maze.get_agent_pos()) < 0
                ):
                    helper.log_error("agent hit obstacle")
                    helper.quit()

            action = self.act(maze.state())
            if action == None:
                helper.log_error("action is None!")
            if verbose:
                print("action: ", action)
            self.steps += 1
            maze.update_visited(action)
            reward = maze.step(action)
            rewards.append(reward)
            # don't learn
        cof = 1
        returns = 0
        for i in range(len(rewards)):
            returns += rewards[i] * cof
            cof *= discount_factor

        if verbose:
            helper.log_success(
                f"agent reach end at {maze.get_agent_pos()} after {self.steps} steps with return {returns}"
            )
        return returns

    def walk(self, maze: Maze, verbose=False):
        self.steps = 0
        discount_factor = self.discount_factor
        rewards = []
        while not maze.is_end():
            if verbose:
                print("agent pos: ", maze.get_agent_pos())
            action = self.act(maze.state())
            if action == None:
                helper.log_error("action is None!")
            if verbose:
                print("action: ", action)
            self.steps += 1
            maze.update_visited(action)
            reward = maze.step(action)
            rewards.append(reward)
            self.learn(maze.state(), action, reward, verbose)

        cof = 1
        returns = 0
        for i in range(len(rewards)):
            returns += rewards[i] * cof
            cof *= discount_factor

        if verbose:
            helper.log_success(
                f"agent reach end at {maze.get_agent_pos()} after {self.steps} steps with return {returns}"
            )
        return returns

    def learn(
        self, state, action, reward, verbose=False
    ):  # TODO check # BUG can't learn at intermediate state
        """
        Update Q-values using the Q-learning algorithm

        Parameters:
        state - current state
        action - action taken
        reward - reward received
        """
        # Get the current Q-value for the state-action pair
        current_q = self.value(state, action)

        # Get the maximum Q-value for the next state
        max_next_q = float("-inf")
        for next_action in self.action_list:
            if state.valid(next_action):
                next_q = self.value(state, next_action)
                if next_q > max_next_q:
                    max_next_q = next_q

        # If there are no valid actions from this state, set max_next_q to 0
        if max_next_q == float("-inf"):
            max_next_q = 0

        # Get the learning rate from settings
        alpha = self.learning_rate

        # Calculate the new Q-value using the Q-learning formula
        new_q = current_q + alpha * (
            reward + self.discount_factor * max_next_q - current_q
        )

        agent_pos = state.get_agent_pos()

        if verbose:
            helper.log_info(
                f"q_table[({agent_pos}, {action}] = new_q = {new_q} = {current_q} + {alpha} * ({reward} + {self.discount_factor} * {max_next_q} - {current_q})"
            )

        # Update the Q-table
        self.q_table[(agent_pos, action)] = new_q
