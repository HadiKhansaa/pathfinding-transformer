import numpy as np
import random
import config # Assuming config.py has ACTIONS and ACTION_COSTS
# import utils # utils.py uses is_valid FROM this file, careful with circular imports if any

class GridEnvironment:
    def __init__(self, size=None, obstacle_density=0.2, grid_data=None, height=None, width=None): # Made size optional
        self.obstacle_density = obstacle_density
        if grid_data is not None:
            self.grid = np.array(grid_data, dtype=np.int8)
            self.height = self.grid.shape[0]
            self.width = self.grid.shape[1]
            self.size = self.height # Keep self.size for legacy compatibility or if primarily square
                                   # but use self.height and self.width for actual checks.
        elif height is not None and width is not None: # For generating random non-square grids
            self.height = height
            self.width = width
            self.grid = self.generate_random_grid() # Will use self.height, self.width
            self.size = self.height # Or max(height,width) or some other convention
        elif size is not None: # Assume square if only size is given
            self.height = size
            self.width = size
            self.grid = self.generate_random_grid()
            self.size = size
        else:
            raise ValueError("Must provide grid_data, or height/width, or size to GridEnvironment")


    def generate_random_grid(self):
        # Uses self.height and self.width
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        num_obstacles = int(self.obstacle_density * self.height * self.width)
        
        # Ensure we don't try to place more obstacles than free cells
        potential_free_cells = list(zip(*np.where(grid == 0)))
        if len(potential_free_cells) <= num_obstacles + 2:
             num_obstacles = max(0, len(potential_free_cells) - 3)
        
        if num_obstacles > 0 and potential_free_cells:
            obstacle_indices_indices = random.sample(range(len(potential_free_cells)), num_obstacles)
            obstacle_indices = [potential_free_cells[i] for i in obstacle_indices_indices]
            for r, c in obstacle_indices:
                grid[r, c] = 1
        return grid

    def generate_maze(self, complexity=0.75, density=0.75):
        # Maze generation might assume square or needs adjustment for non-square
        # For now, let's assume it works best if self.height and self.width are similar
        # Or it primarily uses the smaller dimension for its base shape.
        # This part needs careful review if you intend to generate non-square mazes.
        # The original code used self.size (which was square)
        # For simplicity, let's make it generate based on an effective square size, then crop/pad.
        
        # Using an effective square size for maze generation logic
        effective_size_for_maze = min(self.height, self.width) # Or some other logic
        shape_maze = (effective_size_for_maze // 2 * 2 + 1, effective_size_for_maze // 2 * 2 + 1)

        complexity_val = int(complexity * (5 * (shape_maze[0] + shape_maze[1])))
        density_val = int(density * ((shape_maze[0] // 2) * (shape_maze[1] // 2)))
        
        Z = np.zeros(shape_maze, dtype=bool)
        Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = 1

        # Carving logic (simplified from your original, ensure it's robust)
        # This is a basic random DFS maze generator
        stack = [(random.randint(0, shape_maze[1]//2-1)*2+1, random.randint(0, shape_maze[0]//2-1)*2+1)]
        Z[stack[0][1], stack[0][0]] = 0 # Mark start as path

        while stack:
            x, y = stack[-1]
            neighbours = []
            if x > 1 and Z[y, x - 2]: neighbours.append((x - 2, y, x-1, y)) # nx, ny, wallx, wally
            if x < shape_maze[1] - 2 and Z[y, x + 2]: neighbours.append((x + 2, y, x+1, y))
            if y > 1 and Z[y - 2, x]: neighbours.append((x, y - 2, x, y-1))
            if y < shape_maze[0] - 2 and Z[y + 2, x]: neighbours.append((x, y + 2, x, y+1))

            if neighbours:
                nx, ny, wx, wy = random.choice(neighbours)
                Z[ny, nx] = 0
                Z[wy, wx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        maze_grid_temp = Z.astype(np.int8)

        # Now, fit this generated maze_grid_temp into self.grid (self.height, self.width)
        # This involves padding or cropping. Let's center and pad with obstacles.
        final_grid = np.ones((self.height, self.width), dtype=np.int8) # Start with all obstacles

        r_start_dest = (self.height - maze_grid_temp.shape[0]) // 2
        c_start_dest = (self.width - maze_grid_temp.shape[1]) // 2
        
        r_end_dest = r_start_dest + maze_grid_temp.shape[0]
        c_end_dest = c_start_dest + maze_grid_temp.shape[1]

        # Ensure slices are valid
        r_start_src, c_start_src = 0, 0
        r_end_src, c_end_src = maze_grid_temp.shape[0], maze_grid_temp.shape[1]

        # Adjust destination slices if maze is larger than target grid (cropping)
        if r_start_dest < 0: 
            r_start_src = -r_start_dest
            r_start_dest = 0
        if c_start_dest < 0:
            c_start_src = -c_start_dest
            c_start_dest = 0
        
        r_end_dest = min(self.height, r_end_dest)
        c_end_dest = min(self.width, c_end_dest)

        r_len_copy = min(maze_grid_temp.shape[0]-r_start_src, r_end_dest-r_start_dest)
        c_len_copy = min(maze_grid_temp.shape[1]-c_start_src, c_end_dest-c_start_dest)

        if r_len_copy > 0 and c_len_copy > 0 :
            final_grid[r_start_dest : r_start_dest+r_len_copy, c_start_dest : c_start_dest+c_len_copy] = \
                maze_grid_temp[r_start_src : r_start_src+r_len_copy, c_start_src : c_start_src+c_len_copy]
        
        self.grid = final_grid
        return self.grid


    def get_neighbors(self, r, c):
        neighbors = []
        for i, (dr, dc) in enumerate(config.ACTIONS):
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc) and not self.is_obstacle(nr, nc): # is_obstacle will use correct self.height/width
                cost = config.ACTION_COSTS[i]
                neighbors.append(((nr, nc), cost, i))
        return neighbors

    def is_valid(self, r, c):
        """Checks if coordinates are within grid bounds using self.height and self.width."""
        return 0 <= r < self.height and 0 <= c < self.width

    def is_obstacle(self, r, c):
        """Checks if a cell is an obstacle or out of bounds."""
        if not self.is_valid(r, c): # This check now uses self.height and self.width
            return True
        return self.grid[r, c] == 1 # This should now be safe

    def find_random_free_cell(self):
        """Finds a random non-obstacle cell."""
        free_cells = list(zip(*np.where(self.grid == 0)))
        if not free_cells:
            return None
        return random.choice(free_cells)

    def get_random_start_goal_pair(self, min_dist_factor=0.1):
        """Finds a random pair of non-obstacle start and goal cells."""
        min_dist = max(1, int(min(self.height, self.width) * min_dist_factor))
        start = self.find_random_free_cell()
        goal = self.find_random_free_cell()
        attempts = 0
        while goal is None or start is None or start == goal or \
              np.linalg.norm(np.array(start) - np.array(goal)) < min_dist:
            if attempts > 100:
                # print("Warning: Could not find suitable start/goal pair after 100 attempts.")
                return None, None
            start = self.find_random_free_cell()
            goal = self.find_random_free_cell()
            attempts += 1
        return start, goal