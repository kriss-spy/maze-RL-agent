import helper
import env
import agent
import train


def main(
    default_maze_matrix=[
        ["b", 0, 0, 0],
        [0, -1, 0, -1],
        [0, 0, 0, -1],
        [-1, 0, 0, "e"],
    ]
):
    # Initialize default maze

    # default_maze_matrix = [["e", -1, -2], [0, -3, -4], [0, 0, "b"]]

    # Settings dictionary
    default_settings = {
        "learning_rate": 0.1,  # alpha
        "discount_factor": 0.9,  # gamma
        "exploration_rate": 1.0,  # epsilon - start high
        "exploration_decay_rate": 0.995,  # Epsilon decay rate per episode
        "min_exploration_rate": 0.01,  # Minimum epsilon
        "episodes": 1000,
        "batch_size": 100,  # Not currently used in the Q-learning logic directly
        "default_value": 0,  # Default Q-value for unseen state-action pairs
        "end_reward": 10,
        "step_cost": -0.5,
    }

    settings = default_settings
    maze_matrix = default_maze_matrix

    maze = env.Maze(maze_matrix, settings)
    maze_agent = agent.Agent(settings)

    # Display welcome message
    helper.print_welcome()

    # Main program loop
    running = True
    while running:
        # Get user input
        helper.console.print("[dim]Enter command (h for help):[/]", end=" ")
        key = input().strip().lower()

        if key == "h":
            # Help - show available commands
            helper.print_hotkeys()

        elif key == "s":
            # Show current settings
            helper.print_header("Current Settings")
            for setting, value in settings.items():
                helper.console.print(f"[cyan]{setting}:[/] {value}")

            helper.print_maze_matrix(maze.get_matrix())

        elif key == "r":
            # Reset settings to default
            settings = default_settings
            maze_matrix = default_maze_matrix
            helper.log_success("Settings reset to default values")
            maze = env.Maze(maze_matrix, settings)
            maze_agent = agent.Agent(settings)

        elif key == "a":
            # Auto run all steps
            helper.log_info("Auto-running all steps...")

            helper.print_header("Training Agent")
            train.train_agent(maze, maze_agent, settings)

            helper.print_header("Agent Navigation")
            maze = env.Maze(maze_matrix, settings)
            maze_agent.inf_walk(maze)

        elif key == "m":
            # Input customized maze
            maze_matrix = helper.input_maze()
            maze = env.Maze(maze_matrix, settings)
            helper.log_success("Custom maze created and set")

        elif key == "t":
            # Train agent
            helper.print_header("Training Agent")
            train.train_agent(maze, maze_agent, settings)

        elif key == "w":
            # Run one turn with learning
            helper.print_header("Agent Walk")
            maze = env.Maze(maze_matrix, settings)
            maze_agent.walk(maze, verbose=True)

        elif key == "i":
            # Show agent navigating the maze
            helper.print_header("Agent Navigation")
            maze = env.Maze(maze_matrix, settings)
            maze_agent.inf_walk(maze)

        elif key == "q":
            # Quit program
            running = False

        else:
            helper.log_warning(f"Unknown command: '{key}'. Press 'h' for help.")

    helper.quit()


def test():
    test_case_1 = [
        ["b", 0, -1, 0],
        [0, -1, 1, 0],
        [0, 2, 0, 0],
        [-1, 0, 0, "e"],
    ]
    main(test_case_1)

    test_case_2 = [
        ["b", -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 2, -1],
        [0, 0, 0, "e"],
    ]
    main(test_case_2)

    test_case_3 = [
        ["b", 1, 0, -1],
        [0, -1, -1, 0],
        [0, 2, 0, 0],
        [2, 0, -1, "e"],
    ]
    main(test_case_3)
    test_case_4 = [
        ["b", 0, 0, 0],
        [-1, 1, 0, 2],
        [0, -1, -1, 0],
        [0, 0, 0, "e"],
    ]
    main(test_case_4)


if __name__ == "__main__":  # Fixed the condition
    main()
    # test()
