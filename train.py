import helper


def train_agent(maze, agent, settings):
    helper.print_header("Training Start")
    batch_size = settings["batch_size"]
    episodes = settings["episodes"]
    for epoch in range(episodes):
        returns = agent.walk(maze, settings, verbose=False)
        if epoch % batch_size:
            helper.log_info(f"[{epoch}/{episodes}]: {returns}")
    helper.log_success("train finished!")
