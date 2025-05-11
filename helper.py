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
    hotkeys_table.add_row("d", "Show agent navigating the maze")
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
