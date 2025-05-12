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
        self.action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.q_table = {}
        self.exploration_rate = settings["exploration_rate"]
        self.initial_exploration_rate = settings["exploration_rate"]  # For decay
        self.exploration_decay_rate = settings.get(
            "exploration_decay_rate", 0.995
        )  # Default decay rate
        self.min_exploration_rate = settings.get(
            "min_exploration_rate", 0.01
        )  # Default min exploration
        self.default_value = settings["default_value"]
        self.discount_factor = settings["discount_factor"]
        self.learning_rate = settings["learning_rate"]
        self.steps = 0

    def value(self, pos, action):  # Changed state to pos for clarity
        # Use agent position as the state representation in q_table instead of Maze object
        if (pos, action) in self.q_table.keys():
            return self.q_table[(pos, action)]
        else:
            self.q_table[(pos, action)] = self.default_value
            return self.default_value

    def act(self, maze_state: Maze):  # Parameter type hint for clarity
        current_pos = maze_state.get_agent_pos()
        agent_act = None
        rn = random.random()
        best_act = None
        max_val = float("-inf")

        # Use maze_state.is_valid_move for checking actions from current_pos
        valid_action_list = [
            act
            for act in self.action_list
            if maze_state.is_valid_move(current_pos, act)
        ]

        if not valid_action_list:
            # This case should ideally not happen if the maze is solvable and agent is not trapped.
            # If it does, it means there are no valid moves from the current position.
            # For now, we'll return None, but a more robust solution might be needed
            # depending on how terminal states / inescapable traps are handled.
            # helper.log_warning(f"No valid actions from {current_pos}")
            return None  # Or a random action from self.action_list if we want to allow bumping into walls

        if rn > self.exploration_rate:  # exploit
            for action in valid_action_list:
                val = self.value(current_pos, action)  # Use current_pos
                if val > max_val:
                    best_act = action
                    max_val = val
            agent_act = best_act
            # If multiple actions have the same max_val, pick one randomly
            if (
                agent_act is None and valid_action_list
            ):  # Should only happen if all q-values are -inf or default
                best_actions = [
                    act
                    for act in valid_action_list
                    if self.value(current_pos, act) == max_val
                ]
                if best_actions:
                    agent_act = random.choice(best_actions)
                else:  # Fallback if something unexpected happens
                    agent_act = random.choice(valid_action_list)

        else:  # explore
            agent_act = random.choice(valid_action_list)

        if agent_act is None and valid_action_list:  # Final fallback
            agent_act = random.choice(valid_action_list)

        return agent_act

    def reverse_act(self, action):
        return (
            0 if action[0] == 0 else -1 * action[0],
            0 if action[1] == 0 else -1 * action[1],
        )

    def inf_walk(self, maze: Maze, verbose=True):
        self.steps = 0
        # discount_factor = self.discount_factor # Not used for returns calculation in inf_walk
        # rewards = [] # Not used
        maze.agent_pos = maze.get_begin()  # Reset agent position to start

        while not maze.is_end():
            current_pos = maze.get_agent_pos()
            if verbose:
                print(f"agent pos: {current_pos} [{maze.at(current_pos)}]")
                # Obstacle check is implicitly handled by act and step,
                # but an explicit check here might be desired for specific logging.
                # if maze.is_obstacle(current_pos) and current_pos not in [maze.get_begin(), maze.get_end()]:
                #     helper.log_error(f"Agent somehow moved into an obstacle at {current_pos}")
                #     helper.quit() # Or handle error appropriately

            action = self.act(maze)  # Pass the whole maze object as state
            if action is None:
                helper.log_error(
                    f"No action possible from {current_pos}. Agent is stuck or maze is invalid."
                )
                # Decide how to handle this: break, quit, or specific return
                return -float("inf")  # Indicate failure

            if verbose:
                print("action: ", action)

            self.steps += 1
            # maze.update_visited(action) # Removed, no longer tracking visited in this way

            # reward = maze.step(action) # Step updates agent_pos and returns reward
            maze.step(action)  # Agent position is updated within maze.step
            # rewards.append(reward) # Not needed for inf_walk as it doesn't calculate returns

            if (
                self.steps > (len(maze.maze_matrix) * len(maze.maze_matrix[0])) * 4
            ):  # Safety break for very long walks
                helper.log_warning("Agent taking too long, possibly stuck in a loop.")
                return -float("inf")  # Indicate failure due to excessive steps

        # Returns calculation was present but inf_walk doesn't learn, so Q-values aren't updated.
        # The original return calculation was for the episode's return, which is fine to keep if desired.
        # However, for pure inference (inf_walk), the number of steps is a more direct performance metric.
        # For consistency with `walk`, let's calculate and return discounted rewards.
        # To do this properly, we'd need to store rewards during the walk.
        # For now, let's simplify and just report success and steps.
        # The original code did calculate returns, so let's re-enable it if we track rewards.
        # For now, inf_walk will not calculate returns, just show path.
        # If returns are needed, `walk` should be used or `inf_walk` modified to track rewards.

        if verbose:
            helper.log_success(
                f"Agent reached end at {maze.get_agent_pos()} after {self.steps} steps."
            )
        return self.steps  # Returning steps as a measure of performance for inference

    def walk(self, maze: Maze, verbose=False):
        self.steps = 0
        # discount_factor = self.discount_factor # Already an instance variable
        rewards_log = []  # Log of rewards received in the episode
        maze.agent_pos = maze.get_begin()  # Ensure agent starts at the beginning

        while not maze.is_end():
            current_pos = maze.get_agent_pos()
            if verbose:
                print("agent pos: ", current_pos)

            action = self.act(maze)  # Pass the whole maze object as state

            if action is None:
                helper.log_warning(
                    f"No action possible from {current_pos} in walk. Episode terminated."
                )
                # This state might occur if the agent gets into a position with no valid moves.
                # Penalize heavily or handle as a failed episode.
                # For Q-learning, we might not update Q-value here or assign a large negative reward.
                break  # End episode if stuck

            if verbose:
                print("action: ", action)

            self.steps += 1
            # maze.update_visited(action) # Removed

            reward = maze.step(action)  # maze.agent_pos is updated here
            rewards_log.append(reward)
            next_pos = maze.get_agent_pos()

            is_terminal = maze.is_end() or maze.is_obstacle(
                next_pos
            )  # Check if next_pos is terminal

            self.learn(
                current_pos, action, reward, next_pos, is_terminal, maze, verbose
            )

            if (
                self.steps > (len(maze.maze_matrix) * len(maze.maze_matrix[0])) * 4
            ):  # Safety break
                if verbose:
                    helper.log_warning("Walk taking too long, episode terminated.")
                break

        # Calculate total discounted return for the episode
        episode_return = 0
        current_discount_factor = 1
        for r in rewards_log:
            episode_return += r * current_discount_factor
            current_discount_factor *= self.discount_factor

        if verbose:
            if maze.is_end():
                helper.log_success(
                    f"Agent reached end at {maze.get_agent_pos()} after {self.steps} steps with return {episode_return:.2f}"
                )
            else:
                helper.log_warning(
                    f"Episode ended after {self.steps} steps without reaching the goal. Return: {episode_return:.2f}"
                )
        return episode_return

    def learn(
        self,
        current_pos,
        action,
        reward,
        next_pos,
        is_terminal,
        maze: Maze,
        verbose=False,
    ):
        """
        Update Q-values using the Q-learning algorithm
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
        """
        # Get the current Q-value for the state-action pair (s, a)
        # current_q = self.value(state, action) # Original, state was s'
        current_q = self.value(current_pos, action)

        # Get the maximum Q-value for the next state (s')
        max_q_next = 0.0  # Default if terminal or no valid actions from next_pos
        if not is_terminal:
            possible_next_actions = [
                act for act in self.action_list if maze.is_valid_move(next_pos, act)
            ]
            if possible_next_actions:
                max_q_next = float("-inf")
                for next_action_candidate in possible_next_actions:
                    q_val = self.value(next_pos, next_action_candidate)
                    if q_val > max_q_next:
                        max_q_next = q_val
            # If no possible_next_actions, max_q_next remains 0.0, which is correct (no future reward)
            if max_q_next == float(
                "-inf"
            ):  # Should not happen if possible_next_actions was populated
                max_q_next = 0.0

        # Get the learning rate
        alpha = self.learning_rate

        # Calculate the new Q-value using the Q-learning formula
        new_q = current_q + alpha * (
            reward + self.discount_factor * max_q_next - current_q
        )

        if verbose:
            helper.log_info(
                f"learn: s={current_pos}, a={action}, r={reward}, s'={next_pos}, terminal={is_terminal}"
            )
            helper.log_info(
                f"  Q({current_pos}, {action}) updated from {current_q:.4f} to {new_q:.4f}"
            )
            helper.log_info(
                f"  alpha={alpha}, reward={reward}, gamma={self.discount_factor}, max_q_next={max_q_next:.4f}"
            )
            helper.log_info(
                f"  new_q = {current_q:.4f} + {alpha} * ({reward} + {self.discount_factor} * {max_q_next:.4f} - {current_q:.4f})"
            )

        # Update the Q-table for (s,a)
        self.q_table[(current_pos, action)] = new_q

    def decay_exploration_rate(self, episode, total_episodes):
        # Simple linear decay, or exponential decay
        # self.exploration_rate = self.initial_exploration_rate - (self.initial_exploration_rate - self.min_exploration_rate) * (episode / total_episodes)
        # Exponential decay:
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay_rate,
        )
