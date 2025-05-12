import numpy as np
import random
import config
import utils # For is_valid

class GridEnvironment:
    def __init__(self, size, obstacle_density=0.2, grid_data=None):
        self.size = size
        self.obstacle_density = obstacle_density
        if grid_data is not None:
            self.grid = np.array(grid_data, dtype=np.int8)
            self.size = self.grid.shape[0] # Assuming square grid
        else:
            self.grid = self.generate_random_grid()

    def generate_random_grid(self):
        grid = np.zeros((self.size, self.size), dtype=np.int8)
        num_obstacles = int(self.obstacle_density * self.size * self.size)
        # Ensure start/goal chosen later are not blocked initially
        potential_free_cells = list(zip(*np.where(grid == 0)))
        if len(potential_free_cells) <= num_obstacles + 2: # Avoid making grid impossible
             num_obstacles = max(0, len(potential_free_cells) - 3)

        obstacle_indices = random.sample(potential_free_cells, num_obstacles)
        for r, c in obstacle_indices:
             grid[r, c] = 1  # 1 represents an obstacle
        return grid

    def generate_maze(self, complexity=0.75, density=0.75):
        # Ensure odd dimensions for standard maze generation algorithms
        shape = (self.size // 2 * 2 + 1, self.size // 2 * 2 + 1)
        original_size = self.size
        self.size = shape[0] # Update size

        # If the requested size was even, we might need to pad later to match TARGET_GRID_SIZE if necessary
        # Or, adjust generation complexity/density based on the new shape
        complexity = int(complexity * (5 * (shape[0] + shape[1]))) # Scale complexity
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) # Scale density

        # Maze generation logic (from original code)
        Z = np.zeros(shape, dtype=bool) # False is path, True is wall
        # Fill borders
        Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = 1 # Wall = 1

        # Start carving
        x, y = (random.randint(0, shape[1] // 2 - 1) * 2 + 1,
                random.randint(0, shape[0] // 2 - 1) * 2 + 1)
        Z[y, x] = 0 # Start cell is path
        stack = [(x, y)]

        while stack:
            x, y = stack[-1]
            neighbours = []
            # Check potential neighbours (2 steps away)
            if x > 1 and Z[y, x - 2]: neighbours.append((x - 2, y))
            if x < shape[1] - 2 and Z[y, x + 2]: neighbours.append((x + 2, y))
            if y > 1 and Z[y - 2, x]: neighbours.append((x, y - 2))
            if y < shape[0] - 2 and Z[y + 2, x]: neighbours.append((x, y + 2))

            if neighbours:
                nx, ny = random.choice(neighbours)
                # Carve path to neighbour
                Z[ny, nx] = 0 # Neighbour cell is path
                Z[y + (ny - y) // 2, x + (nx - x) // 2] = 0 # Wall between is path
                stack.append((nx, ny))
            else:
                stack.pop() # Backtrack

        self.grid = Z.astype(np.int8) # 1 is obstacle (wall), 0 is free path

        # Optional: Add some random openings or obstacles back for variation
        # num_random_flips = int(0.01 * self.size * self.size) # Flip 1%
        # for _ in range(num_random_flips):
        #     r, c = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
        #     self.grid[r,c] = 1 - self.grid[r,c] # Flip state

        # If original requested size was different (e.g., even), resize/pad if needed
        # This is tricky - maybe better to ensure TARGET_GRID_SIZE is odd if using mazes often.
        if self.size != original_size and self.size < config.TARGET_GRID_SIZE:
             # Example: Pad to TARGET_GRID_SIZE (might create unreachable areas)
             pad_width = config.TARGET_GRID_SIZE - self.size
             pad_t = pad_width // 2
             pad_b = pad_width - pad_t
             self.grid = np.pad(self.grid, ((pad_t, pad_b), (pad_t, pad_b)), mode='constant', constant_values=1)
             self.size = config.TARGET_GRID_SIZE
        elif self.size > config.TARGET_GRID_SIZE:
             # Crop (center crop)
             crop_start = (self.size - config.TARGET_GRID_SIZE) // 2
             crop_end = crop_start + config.TARGET_GRID_SIZE
             self.grid = self.grid[crop_start:crop_end, crop_start:crop_end]
             self.size = config.TARGET_GRID_SIZE

        return self.grid


    def get_neighbors(self, r, c):
        neighbors = []
        for i, (dr, dc) in enumerate(config.ACTIONS):
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc) and not self.is_obstacle(nr, nc):
                cost = config.ACTION_COSTS[i]
                neighbors.append(((nr, nc), cost, i)) # node, cost_to_reach, action_index
        return neighbors

    def is_valid(self, r, c):
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.size and 0 <= c < self.size

    def is_obstacle(self, r, c):
        """Checks if a cell is an obstacle or out of bounds."""
        if not self.is_valid(r, c):
            return True # Out of bounds is treated as an obstacle
        return self.grid[r, c] == 1

    def find_random_free_cell(self):
        """Finds a random non-obstacle cell."""
        free_cells = list(zip(*np.where(self.grid == 0)))
        if not free_cells:
            return None # No free cells
        return random.choice(free_cells)

    def get_random_start_goal_pair(self, min_dist=1):
        """Finds a random pair of non-obstacle start and goal cells."""
        start = self.find_random_free_cell()
        goal = self.find_random_free_cell()
        attempts = 0
        while goal is None or start is None or start == goal or np.linalg.norm(np.array(start) - np.array(goal)) < min_dist:
            if attempts > 100: # Prevent infinite loop on sparse grids
                print("Warning: Could not find suitable start/goal pair after 100 attempts.")
                return None, None
            start = self.find_random_free_cell()
            goal = self.find_random_free_cell()
            attempts += 1
        return start, goal
