import matplotlib.pyplot as plt
import numpy as np

# Sample points
points = [
    (1.5, 2.5, 3.0), (1.7, 2.4, 2.8), (2.1, 3.0, 4.2),
    (1.0, 2.0, 1.5), (3.2, 3.7, 2.0), (2.5, 1.8, 3.5)
]

# Extract x, y, z coordinates
x_vals, y_vals, z_vals = zip(*points)

# Define grid resolution
grid_resolution = 0.5

# Calculate grid edges for X and Y
x_min, x_max = min(x_vals), max(x_vals)
y_min, y_max = min(y_vals), max(y_vals)
x_edges = np.arange(x_min, x_max + grid_resolution, grid_resolution)
y_edges = np.arange(y_min, y_max + grid_resolution, grid_resolution)

# Create a 2D grid for visualization
X, Y = np.meshgrid(
    (x_edges[:-1] + x_edges[1:]) / 2,  # Midpoints of x bins
    (y_edges[:-1] + y_edges[1:]) / 2   # Midpoints of y bins
)

# Initialize Z grid with NaNs
Z = np.full(X.shape, np.nan)

# Fill Z grid based on the points
for x, y, z in points:
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1
    if 0 <= x_idx < X.shape[1] and 0 <= y_idx < X.shape[0]:
        Z[y_idx, x_idx] = z  # Assign z value to the corresponding grid cell

# Plot the grid
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', edgecolors='k', linewidth=0.5)
plt.colorbar(label='Z Value')
plt.scatter(x_vals, y_vals, c=z_vals, cmap='viridis', edgecolor='white', s=100, label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grid Visualization')
plt.legend()
plt.show()
