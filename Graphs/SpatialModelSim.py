import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# Model parameters
GRID_SIZE = 50
INITIAL_INFECTED_COUNT = 5
INFECTION_PROBABILITY = 0.3
RECOVERY_PROBABILITY = 0.1
SIMULATION_DAYS = 100

# States
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
initial_infected_indices = np.random.choice(GRID_SIZE * GRID_SIZE, INITIAL_INFECTED_COUNT, replace=False)
for idx in initial_infected_indices:
    row, col = divmod(idx, GRID_SIZE)
    grid[row, col] = INFECTED

# Set up plot
cmap = ListedColormap(['blue', 'red', 'red'])
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grid, cmap=cmap, origin='lower')
ax.set_title("Day 0")

# Simulation update function
def update(day):
    global grid
    new_grid = np.copy(grid)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == INFECTED:
                # Infect neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if grid[nr, nc] == SUSCEPTIBLE and np.random.rand() < INFECTION_PROBABILITY:
                                new_grid[nr, nc] = INFECTED
                # Recover
                if np.random.rand() < RECOVERY_PROBABILITY:
                    new_grid[r, c] = RECOVERED

    grid = new_grid
    im.set_data(grid)
    ax.set_title(f"Day {day + 1}")
    return [im]

# Create looping animation
anim = FuncAnimation(
    fig,
    update,
    frames=SIMULATION_DAYS,
    interval=200,
    blit=True,
    repeat=True 
)

plt.show()
