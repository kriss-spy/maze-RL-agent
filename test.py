import random
import rich
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel

# Define custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
    }
)

console = Console(theme=custom_theme)


def log_info(message):
    """Log information messages"""
    console.print(f"ℹ️ [info]{message}[/]")


def log_success(message):
    """Log success messages"""
    console.print(f"✅ [success]{message}[/]")


def log_warning(message):
    """Log warning messages"""
    console.print(f"⚠️ [warning]{message}[/]")


def log_error(message):
    """Log error messages"""
    console.print(f"❌ [error]{message}[/]")


def print_header(title):
    """Print a fancy header"""
    console.print(Panel(f"[bold cyan]{title}[/]", expand=False))


def safe_input(prompt, validator=None, error_msg=None, converter=None):
    """
    Get user input with validation.

    Args:
        prompt: The input prompt to display
        validator: Function that returns True if input is valid
        error_msg: Message to display if validation fails
        converter: Function to convert input to desired type

    Returns:
        Validated and possibly converted input
    """
    while True:
        try:
            print(prompt, end="")
            user_input = input().strip()

            # Validate if needed
            if validator and not validator(user_input):
                print(error_msg or "Invalid input. Please try again.")
                continue

            # Convert if needed
            if converter:
                return converter(user_input)
            return user_input

        except ValueError:
            print("Invalid format. Please try again.")
        except KeyboardInterrupt:
            print("\nInput cancelled.")
            quit()
        except Exception as e:
            print(f"An error occurred: {e}")


def print_hotkeys():
    """Display available hotkeys in a styled table"""
    from rich.table import Table

    hotkeys_table = Table(
        title="Available Commands", show_header=True, box=rich.box.ROUNDED
    )

    hotkeys_table.add_column("Key", style="cyan", justify="center")
    hotkeys_table.add_column("Description", style="white")

    hotkeys_table.add_row("h", "Print help")
    hotkeys_table.add_row("s", "Display current settings")
    hotkeys_table.add_row("r", "Reset all settings to default")
    hotkeys_table.add_row("a", "Auto run all steps")
    hotkeys_table.add_row("m", "Input customized maze")
    hotkeys_table.add_row("t", "Train agent")
    hotkeys_table.add_row("w", "Agent walk one turn with learning")
    hotkeys_table.add_row("i", "Show agent navigating the maze")
    hotkeys_table.add_row("q", "Quit program")

    console.print(hotkeys_table)


def print_welcome():
    """Display a welcome message with styling"""
    from rich.align import Align
    from rich.text import Text

    # Create a multiline welcome message with styling
    welcome_text = Text()
    welcome_text.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", style="cyan"
    )
    welcome_text.append(
        "WELCOME TO THE RL MAZE AGENT TRAINING PROGRAM\n", style="bold cyan"
    )
    welcome_text.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n", style="cyan"
    )
    welcome_text.append(
        "Train a reinforcement learning agent to navigate through mazes\n\n",
        style="italic",
    )

    welcome_text.append("» Press ", style="white")
    welcome_text.append("h", style="cyan bold")
    welcome_text.append(" to show help\n", style="white")

    welcome_text.append("» Press ", style="white")
    welcome_text.append("a", style="cyan bold")
    welcome_text.append(" to auto run\n", style="white")

    welcome_text.append("» Press ", style="white")
    welcome_text.append("q", style="cyan bold")
    welcome_text.append(" to quit\n", style="white")

    # Display the welcome message in a panel
    welcome_panel = Panel(
        Align.center(welcome_text),
        border_style="cyan",
        title="RL Maze Agent",
        title_align="center",
    )

    console.print(welcome_panel)


def quit():
    """Exit the program with a farewell message"""
    farewell = Panel(
        "[bold cyan]Thank you for using the RL Maze Agent![/]\n[italic]Goodbye![/]",
        border_style="cyan",
        title="RL Maze Agent",
        title_align="center",
    )
    console.print(farewell)
    exit()


def input_maze():
    # Display header
    print_header("Maze Configuration")

    # Get dimensions with validation
    log_info("Enter maze dimensions (rows columns):")

    def validate_dimensions(input_str):
        parts = input_str.split()
        if len(parts) != 2:
            return False
        return all(p.isdigit() and int(p) > 0 for p in parts)

    def convert_dimensions(input_str):
        return list(map(int, input_str.split()))

    dimensions = safe_input(
        prompt="> ",
        validator=validate_dimensions,
        error_msg="Please enter two positive integers separated by space.",
        converter=convert_dimensions,
    )
    m, n = dimensions[0], dimensions[1]

    log_info(f"Creating a {m}×{n} maze. Enter values for each row:")
    console.print(
        f"[dim](Use integers. Negative for obstacles, positive for rewards, 0 for neutral cells)[/]"
    )

    maze = [[0 for i in range(n)] for j in range(m)]

    # Get each row with validation
    for i in range(m):

        def validate_row(input_str):
            try:
                values = input_str.split()
                if len(values) < n:
                    return False
                return all(
                    v.isdigit() or (v[0] == "-" and v[1:].isdigit()) for v in values[:n]
                )
            except:
                return False

        def convert_row(input_str):
            return list(map(int, input_str.split()))[:n]

        maze[i] = safe_input(
            prompt=f"Row {i+1}/{m} > ",
            validator=validate_row,
            error_msg=f"Please enter at least {n} integers separated by spaces.",
            converter=convert_row,
        )

    log_success("Maze successfully created!")
    # Get begin position with validation
    log_info("Enter starting position (row column):")

    def validate_position(input_str):
        parts = input_str.split()
        if len(parts) != 2:
            return False
        try:
            row, col = map(int, parts)
            return maze[row][col] == 0 and 0 <= row < m and 0 <= col < n
        except ValueError:
            return False

    def convert_position(input_str):
        return list(map(int, input_str.split()))

    begin_pos = safe_input(
        prompt="> ",
        validator=validate_position,
        error_msg=f"Please enter two integers within maze boundaries (0-{m-1} for rows, 0-{n-1} for columns).",
        converter=convert_position,
    )

    # Get end position with validation
    log_info("Enter ending position (row column):")
    end_pos = safe_input(
        prompt="> ",
        validator=validate_position,
        error_msg=f"Please enter two integers within maze boundaries (0-{m-1} for rows, 0-{n-1} for columns).",
        converter=convert_position,
    )

    # Set the begin and end positions in the maze
    begin_row, begin_col = begin_pos
    end_row, end_col = end_pos

    maze[begin_row][begin_col] = "b"
    maze[end_row][end_col] = "e"

    log_success("Start and end positions set successfully.")

    print_maze_matrix(maze)

    return maze


def print_maze_matrix(maze):
    # Create a visual representation of the maze
    from rich.table import Table

    table = Table(title="Maze Configuration", show_header=False, box=rich.box.MINIMAL)

    # Add columns
    for _ in range(len(maze[0])):
        table.add_column()

    # Add rows
    for row in maze:
        styled_cells = []
        for cell in row:
            if cell == "b" or "e":
                styled_cells.append(f"[white]{cell}[/]")
            elif cell < 0:
                # Obstacle
                styled_cells.append(f"[bold red]{cell}[/]")
            elif cell > 0:
                # Reward
                styled_cells.append(f"[bold green]{cell}[/]")
            else:
                # Neutral
                styled_cells.append(f"[white]{cell}[/]")
        table.add_row(*styled_cells)

    console.print(table)


def print_q_table(q_table):
    """Display the Q-table in a styled table format"""
    from rich.table import Table

    # Create a table
    table = Table(title="Q-Table Values", show_header=True, box=rich.box.ROUNDED)

    # Define action names for display
    action_names = {(-1, 0): "Up", (0, 1): "Right", (1, 0): "Down", (0, -1): "Left"}

    # Add columns for state and actions
    table.add_column("State", style="cyan", justify="center")
    table.add_column("Up", justify="right")
    table.add_column("Right", justify="right")
    table.add_column("Down", justify="right")
    table.add_column("Left", justify="right")

    # Group entries by state position
    states = {}
    for (pos, action), value in q_table.items():
        if pos not in states:
            states[pos] = {}
        states[pos][action] = value

    # No entries case
    if not states:
        console.print(table)
        return

    # Process each state
    for pos in sorted(states.keys()):
        actions = states[pos]
        pos_str = f"({pos[0]},{pos[1]})"

        # Values for each direction
        up_val = actions.get((-1, 0), 0.0)
        right_val = actions.get((0, 1), 0.0)
        down_val = actions.get((1, 0), 0.0)
        left_val = actions.get((0, -1), 0.0)

        values = [up_val, right_val, down_val, left_val]

        # Find maximum Q-value
        max_q = max(values)

        # Format values with highlighting
        formatted_values = []
        for value in values:
            if value == max_q and value > 0:
                formatted_values.append(f"[bold green]{value:.4f}[/]")
            else:
                formatted_values.append(f"{value:.4f}")

        # Add the row
        table.add_row(pos_str, *formatted_values)

    # Print the table
    console.print(table)


class Maze:
    maze_matrix = []
    begin_pos = ()
    end_pos = ()
    agent_pos = ()
    end_reward = 0
    step_cost = 0
    visited_set = set()
    visited_stack = []

    def __init__(self, matrix, settings):
        self.maze_matrix = matrix
        self.begin_pos = self.find_begin()
        self.end_pos = self.find_end()
        self.agent_pos = self.begin_pos
        self.end_reward = settings["end_reward"]
        self.step_cost = settings["step_cost"]
        self.visited_set = set()
        self.visited_stack = []
        # Add the beginning position to visited set (not hardcoded to 0,0)
        self.visited_set.add(self.begin_pos)
        self.visited_stack.append(self.begin_pos)

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
        reward = self.end_reward if foo == "e" else self.step_cost
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
                    log_error("agent hit obstacle")
                    quit()

            action = self.act(maze.state())
            if action == None:
                log_error("action is None!")
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
            log_success(
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
                log_error("action is None!")
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
            log_success(
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
            log_info(
                f"q_table[({agent_pos}, {action}] = new_q = {new_q} = {current_q} + {alpha} * ({reward} + {self.discount_factor} * {max_next_q} - {current_q})"
            )

        # Update the Q-table
        self.q_table[(agent_pos, action)] = new_q


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
    log_info(f"Training agent for {episodes} episodes...")

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
            log_info(
                f"Episode {episode + 1}/{episodes} - Avg returns: {avg_returns:.2f}, Best: {best_returns:.2f}"
            )
            print_q_table(maze_agent.q_table)

    log_success(
        f"Training complete! Final avg returns: {avg_returns:.2f}, Best: {best_returns:.2f}"
    )
    # Return the trained agent
    return maze_agent


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
        "exploration_rate": 0.5,  # epsilon
        "episodes": 1000,
        "batch_size": 100,
        "default_value": 0,
        "end_reward": 10,
        "step_cost": -0.5,
    }

    settings = default_settings
    maze_matrix = default_maze_matrix

    maze = Maze(maze_matrix, settings)
    maze_agent = Agent(settings)

    # Display welcome message
    print_welcome()

    # Main program loop
    running = True
    while running:
        # Get user input
        console.print("[dim]Enter command (h for help):[/]", end=" ")
        key = input().strip().lower()

        if key == "h":
            # Help - show available commands
            print_hotkeys()

        elif key == "s":
            # Show current settings
            print_header("Current Settings")
            for setting, value in settings.items():
                console.print(f"[cyan]{setting}:[/] {value}")

            print_maze_matrix(maze.get_matrix())

        elif key == "r":
            # Reset settings to default
            settings = default_settings
            maze_matrix = default_maze_matrix
            log_success("Settings reset to default values")
            maze = Maze(maze_matrix, settings)
            maze_agent = Agent(settings)

        elif key == "a":
            # Auto run all steps
            log_info("Auto-running all steps...")

            print_header("Training Agent")
            train_agent(maze, maze_agent, settings)

            print_header("Agent Navigation")
            maze = Maze(maze_matrix, settings)
            maze_agent.inf_walk(maze)

        elif key == "m":
            # Input customized maze
            maze_matrix = input_maze()
            maze = Maze(maze_matrix, settings)
            log_success("Custom maze created and set")

        elif key == "t":
            # Train agent
            print_header("Training Agent")
            train_agent(maze, maze_agent, settings)

        elif key == "w":
            # Run one turn with learning
            print_header("Agent Walk")
            maze = Maze(maze_matrix, settings)
            maze_agent.walk(maze, verbose=True)

        elif key == "i":
            # Show agent navigating the maze
            print_header("Agent Navigation")
            maze = Maze(maze_matrix, settings)
            maze_agent.inf_walk(maze)

        elif key == "q":
            # Quit program
            running = False

        else:
            log_warning(f"Unknown command: '{key}'. Press 'h' for help.")

    quit()


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
