import helper
import random


class Agent:
    action_list = []
    exploration_rate = 0
    q_table = {}
    default_value = 0

    def __init__(self, settings):
        self.action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.q_table = {}
        self.exploration_rate = settings["exploration_rate"]
        self.default_value = settings["default_value"]

    def value(self, state, action):
        agent_pos = state.get_agent_pos()
        if (state, action) in self.q_table.keys():
            return self.q_table[(state, action)]
        else:
            self.q_table[(state, action)] = self.default_value

    def act(self, state):
        rn = random.random()
        best_act = None
        max_val = 0
        if rn > self.exploration_rate:  # exploit
            for action in self.action_list:
                val = self.value(state, action)
                if (val) > max_val:
                    best_act = action
                    max_val = val
            return best_act
        else:
            return random.choice(self.action_list)  # TODO should include exploit act?

    def inf_walk(self, maze, verbose=True):
        while not maze.is_end():
            if verbose:
                print("agent pos: ", maze.get_agent_pos())
            action = self.act(maze.state())
            reward = maze.step(action)  # don't use reward
            if verbose:
                print("action: ", action)
        if verbose:
            print("agent reach end at ", maze.get_agent_pos())

    def walk(self, maze, settings, verbose=False):
        discount_factor = settings["discount_factor"]
        rewards = []
        while not maze.is_end():
            if verbose:
                print("agent pos: ", maze.get_agent_pos())
            action = self.act(maze.state())
            reward = maze.step(action)
            rewards.append(reward)
            self.learn(maze.state(), action, reward)
            if verbose:
                print("action: ", action)
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
