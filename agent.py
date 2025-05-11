import helper
import random


# TODO define agent
class Agent:
    action_list = []
    exploration_rate = 0

    def __init__(self, exploration_rate):
        self.action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.exploration_rate = exploration_rate

    def act(self, env):
        # TODO implement Agent.act
        rn = random.random()
        if rn > self.exploration_rate:
            pass  # TODO exploit act
        else:
            return random.choice(self.action_list)  # TODO should include exploit act?

    def walk(self, maze):
        while not maze.is_end():
            print("agent pos: ", maze.get_agent_pos())
            action = self.act(maze, self.pos)
            reward = maze.step(action)
            print("action: ", action)

        print("agent reach end at ", maze.get_agent_pos())

    def learn(self, env, action, reward):
        # TODO implement Agent.learn
        pass
