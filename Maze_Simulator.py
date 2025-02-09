import random
import matplotlib.pyplot as plt
import numpy as np

# Do not change the maze generation

class MazeSimulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[1 for _ in range(width)] for _ in range(height)]  # 1 = wall, 0 = path
        self.start = None
        self.end = None

    def generate_maze(self):
        def is_valid_move(x, y):
            return 0 <= x < self.height and 0 <= y < self.width and self.maze[x][y] == 1

        def dfs(x, y):
            self.maze[x][y] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid_move(nx, ny):
                    self.maze[x + dx // 2][y + dy // 2] = 0  # Remove the wall
                    dfs(nx, ny)

        start_x, start_y = random.randrange(0, self.height, 2), random.randrange(0, self.width, 2)
        dfs(start_x, start_y)

    def set_start_and_end(self):
        open_cells = [(x, y) for x in range(self.height) for y in range(self.width) if self.maze[x][y] == 0]
        self.start, self.end = random.sample(open_cells, 2)

    def get_maze_array(self):
        maze_array = [[1 if self.maze[x][y] == 0 or (x, y) in [self.start, self.end] else 0 for y in range(self.width)] for x in range(self.height)]
        return maze_array

    def save_maze_image(self, filename="maze.png"):
        maze_copy = [[2 if (x, y) == self.start else 3 if (x, y) == self.end else self.maze[x][y] for y in range(self.width)] for x in range(self.height)]
        plt.figure(figsize=(10, 10))
        plt.imshow(maze_copy, cmap="viridis", origin="upper")
        plt.axis("off")
        plt.savefig(filename)
        plt.close()


    def mazeSolverUsingRL(self):
        """
        Solve the maze using Q-learning.
        
        The RL agent learns a policy for moving from the start to the end cell.
        Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left.
        Rewards:
        - +100 for reaching the end.
        - -10 for hitting a wall (invalid move).
        - -1 per step to encourage shorter paths.
        
        Returns:
            path (list of tuple): The sequence of (row, col) states representing the optimal path.
        """
        
        # Hyperparameters
        alpha = 0.1        # Learning rate
        gamma = 0.9        # Discount factor
        epsilon = 0.1      # Exploration rate
        num_episodes = 100000  # Number of training episodes
        max_steps = self.width * self.height * 4  # Maximum steps per episode (safeguard)

        # Initialize the Q-table with zeros.
        # Dimensions: [height][width][number_of_actions]
        q_table = np.zeros((self.height, self.width, 4))

        # Define the four possible actions as (dx, dy) movements.
        actions = [(-1, 0),  # Up
                (0, 1),   # Right
                (1, 0),   # Down
                (0, -1)]  # Left

        def is_valid_state(x, y):
            """Return True if (x, y) is within bounds and is an open cell (path)."""
            return 0 <= x < self.height and 0 <= y < self.width and self.maze[x][y] == 0

        def choose_action(state):
            """Choose an action using an epsilon-greedy strategy."""
            if random.uniform(0, 1) < epsilon:
                return random.choice(range(4))  # Explore: choose a random action
            else:
                return np.argmax(q_table[state[0], state[1]])  # Exploit: choose the best-known action

        def get_reward(state):
            """
            Return the reward for moving into a given state.
            - 100 if the state is the end.
            - -10 if the state is invalid (e.g. a wall or out-of-bounds).
            - -1 for any other (valid) step.
            """
            if state == self.end:
                return 100
            if not is_valid_state(state[0], state[1]):
                return -10
            return -1

        # ---------------------------
        # Q-learning Training Loop
        # ---------------------------
        for episode in range(num_episodes):
            state = self.start
            steps = 0
            while steps < max_steps:
                # Choose an action from the current state
                action = choose_action(state)
                x, y = state
                dx, dy = actions[action]
                new_state = (x + dx, y + dy)
                reward = get_reward(new_state)

                # Determine the maximum Q-value for the new state (if valid)
                if is_valid_state(new_state[0], new_state[1]):
                    best_next_q = np.max(q_table[new_state[0], new_state[1]])
                else:
                    best_next_q = 0

                # Update the Q-value for the current state and action
                q_table[x, y, action] += alpha * (reward + gamma * best_next_q - q_table[x, y, action])

                # Terminate the episode if:
                # - The agent reaches the goal.
                # - The agent takes an invalid move (hits a wall).
                if new_state == self.end or not is_valid_state(new_state[0], new_state[1]):
                    break

                # Otherwise, move to the new state and continue
                state = new_state
                steps += 1

        # --------------------------------
        # Extract the Optimal Path
        # --------------------------------
        path = [self.start]
        current_state = self.start
        visited = set([self.start])  # To help prevent loops

        # Follow the greedy policy derived from the Q-table until the end is reached.
        while current_state != self.end:
            action = np.argmax(q_table[current_state[0], current_state[1]])
            dx, dy = actions[action]
            new_state = (current_state[0] + dx, current_state[1] + dy)

            # If the new state is invalid or we've already visited it, break out to avoid an infinite loop.
            if not is_valid_state(new_state[0], new_state[1]) or new_state in visited:
                break

            path.append(new_state)
            visited.add(new_state)
            current_state = new_state

        return path


# Game flow
if __name__ == "__main__":   
    width, height = 21, 21
    
    # Initialize simulator
    simulator = MazeSimulator(width, height)
    simulator.generate_maze()
    simulator.set_start_and_end()
    
    # Generate and display maze
    maze_array = simulator.get_maze_array()
    print("Maze array:")
    for row in maze_array:
        print(row)
    print(f"Start point: {simulator.start}")
    print(f"End point: {simulator.end}")
    
    # Solve maze using RL
    path = simulator.mazeSolverUsingRL()  # Call the solver
    print(f"\nRL Agent Path ({len(path)} steps):")
    print(path)
    
    # Save visualization with path
    simulator.save_maze_image("maze.png")
    print("\nMaze image saved as 'maze.png'")
