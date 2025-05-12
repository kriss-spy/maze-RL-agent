import helper
import env
import agent
import train


def main():
    # Initialize default maze
    default_maze_matrix = [
        ["b", 0, 0, 1],
        [0, -1, 0, -1],
        [0, 0, 0, 1],
        [-1, 0, 0, "e"],
    ]
    maze = env.Maze(default_maze_matrix, settings)
    maze_agent = agent.Agent(settings)

    # Settings dictionary
    default_settings = {
        "learning_rate": 0.1,  # alpha
        "discount_factor": 0.9,  # gamma
        "exploration_rate": 0.3,  # epsilon
        "episodes": 1000,
        "batch_size": 100,
        "default_value": 0,
        "end_reward": 10,
    }

    settings = default_settings

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
            helper.log_success("Settings reset to default values")
            maze = env.Maze(default_maze_matrix, settings)
            maze_agent = agent.Agent(settings)

        elif key == "a":
            # Auto run all steps
            helper.log_info("Auto-running all steps...")
            # Create fresh maze for auto run
            train.train_agent(maze, agent)
            maze_agent.inf_walk(maze)

        elif key == "m":
            # Input customized maze
            custom_maze_matrix = helper.input_maze()
            maze = env.Maze(custom_maze_matrix)
            helper.log_success("Custom maze created and set")

        elif key == "t":
            # Train agent - placeholder for future implementation
            helper.print_header("Training Agent")
            helper.log_info("Agent training functionality will be implemented later")
            train.train_agent(maze, agent, settings)

        elif key == "d":
            # Show agent navigating the maze - placeholder for future implementation
            helper.print_header("Agent Navigation Demo")
            helper.log_info("Agent navigation demo will be implemented later")
            maze_agent.inf_walk(maze)

        elif key == "q":
            # Quit program
            helper.quit()
            running = False

        else:
            helper.log_warning(f"Unknown command: '{key}'. Press 'h' for help.")


if __name__ == "__main__":  # Fixed the condition
    main()
