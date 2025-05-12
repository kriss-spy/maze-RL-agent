import helper
from env import Maze
from agent import Agent


def train_agent(maze: Maze, maze_agent: Agent, settings):  # TODO check
    """
    Train the agent using Q-learning

    Parameters:
    maze - The maze environment
    agent - The Agent instance
    settings - Optional dictionary of settings
    """
    # Get number of episodes from settings, default to 1000
    episodes = settings["episodes"]

    # Store original maze matrix to reset between episodes
    original_matrix = maze.get_matrix()

    # Training loop
    helper.log_info(f"Training agent for {episodes} episodes...")

    best_returns = float("-inf")
    avg_returns = 0

    for episode in range(episodes):
        # Reset maze to initial state for new episode
        # But keep the same agent instance to preserve the Q-table
        maze = maze.__class__(original_matrix, settings)

        # Let the agent walk through the maze and learn
        returns = maze_agent.walk(maze)

        # Track best performance and calculate average
        if returns > best_returns:
            best_returns = returns

        avg_returns = avg_returns + (returns - avg_returns) / (episode + 1)

        # Print progress periodically
        if (episode + 1) % 100 == 0:
            helper.log_info(
                f"Episode {episode + 1}/{episodes} - Avg returns: {avg_returns:.2f}, Best: {best_returns:.2f}"
            )
            helper.print_q_table(maze_agent.q_table)

    helper.log_success(
        f"Training complete! Final avg returns: {avg_returns:.2f}, Best: {best_returns:.2f}"
    )
    # Return the trained agent
    return maze_agent
