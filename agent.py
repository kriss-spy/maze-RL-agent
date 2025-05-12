import helper
import random


class Agent:
    action_list = []
    q_table = {}
    exploration_rate = 0
    default_value = 0
    discount_factor = 0

    def __init__(self, settings):
        self.action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.q_table = {}
        self.exploration_rate = settings["exploration_rate"]
        self.default_value = settings["default_value"]
        self.discount_factor = settings["discount_factor"]

    def value(self, state, action):
        agent_pos = state.get_agent_pos()
        if (state, action) in self.q_table.keys():
            return self.q_table[(state, action)]
        else:
            self.q_table[(state, action)] = self.default_value
            return self.default_value

    def act(self, state):  # TODO check any bug
        agent_act = None
        rn = random.random()
        best_act = None
        max_val = float("-inf")
        valid_action_list = [act for act in self.action_list if state.valid(act)]
        if len(valid_action_list) == 0:
            while True:
                last_visited = state.visited_stack.pop()
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

    def inf_walk(self, maze, verbose=True):
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
            maze.update_visited(action)
            reward = maze.step(action)  # don't use reward
        if verbose:
            print("agent reach end at ", maze.get_agent_pos())

    def walk(self, maze, verbose=False):
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
            maze.update_visited(action)
            reward = maze.step(action)
            rewards.append(reward)
            self.learn(maze.state(), action, reward)
        if verbose:
            print("agent reach end at ", maze.get_agent_pos())
        cof = 1
        returns = 0
        for i in range(len(rewards)):
            returns += rewards[i] * cof
            cof *= discount_factor
        return returns

    def learn(self, state, action, reward):
        # TODO implement Agent.learn
        pass
