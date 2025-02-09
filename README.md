
## Usage

1.  **Clone the Repository:**

    ```
    git clone <repository_url>
    cd maze-simulator
    ```

2.  **Run the Simulator:**

    ```
    python maze_simulator.py
    ```

    This will:

    *   Generate a new maze.
    *   Train the Q-learning agent.
    *   Print the maze array, start point, and end point to the console.
    *   Print the path taken by the agent to the console.
    *   Save a visual representation of the maze and the agent's path to `maze.png`.

## Code Structure

*   `maze_simulator.py`: Contains the main Python script with the `MazeSimulator` class and the Q-learning implementation.

    *   `MazeSimulator` Class:
        *   `__init__(self, width, height)`: Initializes the maze with specified dimensions.
        *   `generate_maze(self)`: Generates the maze using randomized DFS.
        *   `set_start_and_end(self)`: Randomly selects start and end points.
        *   `get_maze_array(self)`: Returns a 2D array representation of the maze.
        *   `save_maze_image(self, filename="maze.png")`: Saves the maze as a PNG image.
        *   `mazeSolverUsingRL(self)`: Implements the Q-learning algorithm to solve the maze.
*   `README.md`: This file (the one you are reading).

## Implementation Details

### Maze Generation

The `generate_maze` method in the `MazeSimulator` class uses a randomized depth-first search algorithm to create the maze. The algorithm starts at a random cell and recursively carves out paths to neighboring cells, ensuring that the maze is fully connected.

### Reinforcement Learning

The `mazeSolverUsingRL` method implements the Q-learning algorithm. Key aspects:

*   **Q-Table:**  A table that stores the expected rewards for taking specific actions in different states.
*   **Epsilon-Greedy Policy:**  A strategy for balancing exploration (trying new actions) and exploitation (using known optimal actions).
*   **Reward Function:**
    *   \+100 for reaching the end point.
    *   -10 for hitting a wall.
    *   -1 for each step taken (to encourage shorter paths).
*   **Hyperparameters:**
    *   `alpha` (learning rate):  Controls how much the agent learns from new experiences.
    *   `gamma` (discount factor):  Determines the importance of future rewards.
    *   `epsilon` (exploration rate):  Controls the balance between exploration and exploitation.
    *   `num_episodes`:  The number of training iterations.

## Customization

You can customize the following parameters:

*   **Maze Dimensions:**  Modify the `width` and `height` variables in the `if __name__ == "__main__":` block to change the size of the maze.  Note: It's recommended to use odd numbers for the dimensions.

    ```
    width, height = 21, 21 # Example:  Change to 31, 31 for a larger maze
    ```

*   **Q-Learning Hyperparameters:**  Adjust the `alpha`, `gamma`, and `epsilon` values in the `mazeSolverUsingRL` function to fine-tune the agent's learning behavior.

    ```
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 100000 # Number of training episodes
    ```

## Future Enhancements

*   Implement other maze-solving algorithms (e.g., A\*, Dijkstra's).
*   Add visualization of the agent's learning process.
*   Experiment with different reward functions.
*   Implement a graphical user interface (GUI) for interactive maze generation and solving.
*   Evaluate the agent's performance on a wider variety of mazes and compare it to other algorithms.

## License

This project is licensed under the [MIT License](LICENSE).  (You should create a `LICENSE` file in your repository with the MIT license text, or choose another license).

## Author

Shivam Chaubey
[shivamchaubey100]
