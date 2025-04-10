# import setup_path
import os, math, time, heapq
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d, norm
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, distance_transform_edt
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import cosysairsim as airsim
from linefit import ground_seg
from function4 import calculate_combined_risks, compute_cvar_cellwise
import cv2

def display_risk_map_cv2(risk_grid, smoothed_path=None, vehicle_pos=None, start=None, dest=None, heading_vector=None):
    vis_grid = np.nan_to_num(risk_grid, nan=0.5)
    norm_grid = (255 * (vis_grid - vis_grid.min()) / (vis_grid.max() - vis_grid.min() + 1e-6)).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_grid, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Convert to 3-channel color image
    overlay = color_map.copy()

    def grid_to_pixel(x, y):
        return int(y), int(x)  # transpose to image format (row, col)

    # Draw smoothed path
    if smoothed_path is not None:
        for i in range(1, len(smoothed_path)):
            pt1 = grid_to_pixel(*smoothed_path[i - 1])
            pt2 = grid_to_pixel(*smoothed_path[i])
            cv2.line(overlay, pt1, pt2, (255, 0, 0), thickness=2)

    # Draw start
    if start is not None:
        cv2.circle(overlay, grid_to_pixel(*start), radius=4, color=(0, 255, 0), thickness=-1)

    # Draw destination
    if dest is not None:
        cv2.circle(overlay, grid_to_pixel(*dest), radius=4, color=(0, 0, 255), thickness=-1)

    # Draw robot position
    if vehicle_pos is not None:
        cv2.circle(overlay, grid_to_pixel(*vehicle_pos), radius=5, color=(255, 255, 255), thickness=-1)

        # Optional: draw heading vector
        if heading_vector is not None:
            x, y = vehicle_pos
            dx, dy = heading_vector
            pt1 = grid_to_pixel(x, y)
            pt2 = grid_to_pixel(x + 3 * dx, y + 3 * dy)
            cv2.arrowedLine(overlay, pt1, pt2, (255, 255, 255), 2, tipLength=0.3)

    resized = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Risk Map View", resized)
    cv2.waitKey(1)

# ---------------------------------------------------------------------------
# Interpolation: Fill in missing (NaN) grid cells using nearby valid cells.
# ---------------------------------------------------------------------------
def interpolate_in_radius(grid, radius):
    """
    Interpolates NaN values in a grid using valid points within a specified radius.
    """
    valid = ~np.isnan(grid)
    valid_coords = np.column_stack(np.where(valid))
    valid_values = grid[valid]
    tree = cKDTree(valid_coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))
    for coord in nan_coords:
        neighbors = tree.query_ball_point(coord, radius)
        if neighbors:
            neighbor_coords = valid_coords[neighbors]
            values = valid_values[neighbors]
            distances = np.linalg.norm(neighbor_coords - coord, axis=1) + 1e-6
            weights = 1.0 / distances
            grid[coord[0], coord[1]] = np.sum(weights * values) / np.sum(weights)
    return grid

# ---------------------------------------------------------------------------
# Helper: Filter points within a given radius.
# ---------------------------------------------------------------------------
def filter_points_by_radius(points, center, radius):
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[distances <= radius]

# ---------------------------------------------------------------------------
# A* Search Helper Functions
# ---------------------------------------------------------------------------
def is_valid(row, col, grid):
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

def is_unblocked(grid, row, col, threshold):
    return (not np.isnan(grid[row, col])) and (grid[row, col] < threshold)

def calculate_h_value(row, col, dest):
    return np.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

def trace_path(cell_details, dest):
    path = []
    row, col = dest
    while True:
        path.append((row, col))
        parent = cell_details[row, col]
        if (row, col) == parent:
            break
        row, col = parent
    path.reverse()
    return path

def a_star_search(risk_grid, start_idx, dest_idx):
    rows, cols = risk_grid.shape
    max_risk = np.nanmax(risk_grid)
    threshold = 0.8 * max_risk if not np.isnan(max_risk) else 6.0

    open_list = []
    heapq.heappush(open_list, (0.0, start_idx))
    g_scores = np.full((rows, cols), np.inf)
    g_scores[start_idx] = 0
    f_scores = np.full((rows, cols), np.inf)
    f_scores[start_idx] = calculate_h_value(*start_idx, dest_idx)
    cell_details = np.full((rows, cols), None, dtype=object)
    for i in range(rows):
        for j in range(cols):
            cell_details[i, j] = (i, j)
    
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == dest_idx:
            return trace_path(cell_details, dest_idx)
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc, risk_grid) and is_unblocked(risk_grid, nr, nc, threshold):
                tentative_g = g_scores[current] + risk_grid[nr, nc]
                if tentative_g < g_scores[nr, nc]:
                    g_scores[nr, nc] = tentative_g
                    f_scores[nr, nc] = tentative_g + calculate_h_value(nr, nc, dest_idx)
                    heapq.heappush(open_list, (f_scores[nr, nc], (nr, nc)))
                    cell_details[nr, nc] = current
    return None

# ---------------------------------------------------------------------------
# Smoothing Function: Smoothens a path using a moving average filter.
# ---------------------------------------------------------------------------
def smooth_path(path, window_size=5):
    """
    Smooths a sequence of (x,y) points using a simple moving average filter.
    """
    path = np.array(path)
    n_points = len(path)
    if n_points < window_size:
        return path
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed = [np.mean(path[max(0, i-half_window):min(n_points, i+half_window+1)], axis=0)
                for i in range(n_points)]
    return np.array(smoothed)

# ---------------------------------------------------------------------------
# Lidar and Vehicle Pose Handling
# ---------------------------------------------------------------------------
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar):
        if gpulidar:
            lidarData = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            lidarData = self.client.getLidarData(self.lidarName, self.vehicleName)
        if lidarData.time_stamp != self.lastlidarTimeStamp:
            self.lastlidarTimeStamp = lidarData.time_stamp
            if len(lidarData.point_cloud) < 2:
                return None, None
            points = np.array(lidarData.point_cloud, dtype=np.float32)
            num_dims = 5 if gpulidar else 3
            points = points.reshape((-1, num_dims))
            if not gpulidar:
                points = points * np.array([1, -1, 1])
            return points, lidarData.time_stamp
        return None, None

    def get_vehicle_pose(self):
        vehicle_pose = self.client.simGetVehiclePose()
        pos = vehicle_pose.position
        orient = vehicle_pose.orientation
        position_array = np.array([float(pos.x_val), float(pos.y_val), float(pos.z_val)])
        rotation_matrix = self.quaternion_to_rotation_matrix(orient)
        return position_array, rotation_matrix

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])

    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position

# ---------------------------------------------------------------------------
# Grid Map: Accumulates ground (and obstacle) heights per cell.
# ---------------------------------------------------------------------------
class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        # Store (sum, count) per cell
        self.grid = {}

    def get_grid_cell(self, x, y):
        return (round(x / self.resolution, 1), round(y / self.resolution, 1))

    def add_point(self, x, y, z, timestamp):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, 1]
        else:
            self.grid[cell][0] += z
            self.grid[cell][1] += 1

    def get_height_estimate(self):
        estimates = []
        for (gx, gy), (z_sum, count) in self.grid.items():
            mean_z = z_sum / count
            estimates.append([gx * self.resolution, gy * self.resolution, mean_z])
        return np.array(estimates)

# ---------------------------------------------------------------------------
# PID Controller for Steering and Forward Motion
# ---------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, dt=0.1):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# ---------------------------------------------------------------------------
# Fade risk near high-risk cells using a distance transform.
# ---------------------------------------------------------------------------
def fade_with_distance_transform(risk_grid, high_threshold=0.4, fade_scale=4.0, sigma=5.0):
    grid_max = np.nanmax(risk_grid)
    threshold_val = high_threshold * grid_max
    high_mask = risk_grid > threshold_val
    dist_map = distance_transform_edt(~high_mask)
    fade_risk = fade_scale * np.exp(-dist_map / sigma)
    return np.maximum(risk_grid, fade_risk)

# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize lidar and grid maps.
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    grid_map_ground = GridMap(resolution=0.1)
    grid_map_obstacle = GridMap(resolution=0.1)

    # Initialize ground segmentation.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    config_path = os.path.join(BASE_DIR, "../assets/config.toml")
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default parameters")
        groundseg = ground_seg()
    else:
        groundseg = ground_seg(config_path)

    # Initialize visualization.
    fig, ax = plt.subplots()
    plt.ion()
    colorbar = None
    path = None
    temp_dest = None
    temp_path = None
    steering_pid = PIDController(kp=0.8462027727540303, ki=0.023914715286008515, kd=0.0939731107200599, dt=0.1)
    forward_pid  = PIDController(kp=0.4, ki=0.05, kd=0.10, dt=0.1)
    current_target_index = 0

    # Define grid boundaries based on vehicle and destination.
    grid_resolution = 0.1
    margin = 4
    position, rotation_matrix = lidar_test.get_vehicle_pose()
    start_point = np.array([position[0], position[1]])
    destination_point = np.array([17, -8])
    min_x = min(start_point[0], destination_point[0]) - margin
    max_x = max(start_point[0], destination_point[0]) + margin
    min_y = min(start_point[1], destination_point[1]) - margin
    max_y = max(start_point[1], destination_point[1]) + margin

    x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    # Midpoints for visualization and grid indexing.
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_mid, y_mid)

    try:
        while True:
            point_cloud_data, timestamp = lidar_test.get_data(gpulidar=True)
            if point_cloud_data is None:
                continue

            # Process point cloud.
            points = np.array(point_cloud_data[:, :3], dtype=np.float64)
            points = points[np.linalg.norm(points, axis=1) > 0.6]
            position, rotation_matrix = lidar_test.get_vehicle_pose()
            points_world = lidar_test.transform_to_world(points, position, rotation_matrix)
            points_world[:, 2] = -points_world[:, 2]  # Adjust Z if needed
            labels = np.array(groundseg.run(points_world))

            # Populate grid maps based on segmentation.
            for i, point in enumerate(points_world):
                x, y, z = point
                if labels[i] == 1:
                    grid_map_ground.add_point(x, y, z, timestamp)
                elif z > -position[2]:
                    grid_map_obstacle.add_point(x, y, z, timestamp)
                else:
                    grid_map_ground.add_point(x, y, z, timestamp)

            ground_points = grid_map_ground.get_height_estimate()
            obstacle_points = grid_map_obstacle.get_height_estimate()

            vehicle_x, vehicle_y = position[0], position[1]
            center = np.array([vehicle_x, vehicle_y])
            # Filter ground points within a specified radius.
            radius_filter = 13
            ground_points = filter_points_by_radius(ground_points, center, radius_filter)
            if ground_points.size == 0:
                continue

            # Build the ground grid using vectorized binning.
            ground_x_vals = ground_points[:, 0]
            ground_y_vals = ground_points[:, 1]
            ground_z_vals = ground_points[:, 2]
            Z_ground, _, _, _ = binned_statistic_2d(
                ground_x_vals, ground_y_vals, ground_z_vals, statistic='mean', bins=[x_edges, y_edges]
            )

            # Calculate risk grids.
            non_nan_indices = np.argwhere(~np.isnan(Z_ground))
            step_risk_grid, slope_risk_grid = calculate_combined_risks(
                Z_ground, non_nan_indices, max_height_diff=0.035, max_slope_degrees=30.0, radius=0.5
            )
            combined_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
            masked_step_risk = np.ma.masked_array(step_risk_grid, mask=combined_mask) * 3.0
            masked_slope_risk = np.ma.masked_array(slope_risk_grid, mask=combined_mask) * 3.0
            sum_grid = np.ma.filled(masked_step_risk, 0) + np.ma.filled(masked_slope_risk, 0)
            both_nan_mask = np.isnan(step_risk_grid) & np.isnan(slope_risk_grid)
            total_risk_grid = np.where(both_nan_mask, np.nan, sum_grid)

            # Incorporate obstacle risk.
            if obstacle_points.size != 0:
                obstacle_points = filter_points_by_radius(obstacle_points, center, radius_filter)
                if obstacle_points.size != 0:
                    obs_x_idx = np.clip(np.digitize(obstacle_points[:, 0], x_edges) - 1, 0, len(x_mid)-1)
                    obs_y_idx = np.clip(np.digitize(obstacle_points[:, 1], y_edges) - 1, 0, len(y_mid)-1)
                    total_risk_grid[obs_x_idx, obs_y_idx] = 4.0

            # Apply fading and transform risk values.
            total_risk_grid = fade_with_distance_transform(total_risk_grid,
                                                           high_threshold=0.65,
                                                           fade_scale=4.0,
                                                           sigma=3.0)
            max_risk = np.nanmax(total_risk_grid)
            threshold = 0.6 * max_risk
            mask = total_risk_grid > threshold
            total_risk_grid[mask] = np.exp(total_risk_grid[mask])
            total_risk_grid = interpolate_in_radius(total_risk_grid, 1.5)
            masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
            cvar_combined_risk = compute_cvar_cellwise(masked_total_risk_grid, alpha=0.5)
            cvar_combined_risk = cvar_combined_risk.filled(0.50)

            # Mask cells far from the vehicle.
            distance_from_vehicle = np.sqrt((X - vehicle_x)**2 + (Y - vehicle_y)**2)
            cvar_combined_risk[distance_from_vehicle.T > 15.0] = np.nan

            # Convert vehicle and destination positions to grid indices.
            start_idx = (np.digitize(vehicle_x, x_edges) - 1, np.digitize(vehicle_y, y_edges) - 1)
            dest_idx = (np.digitize(destination_point[0], x_edges) - 1, np.digitize(destination_point[1], y_edges) - 1)
            dest_idx = (min(max(dest_idx[0], 0), len(x_mid)-1), min(max(dest_idx[1], 0), len(y_mid)-1))

            # Plan a path if the start cell is valid.
            update_path = False
            if np.isnan(cvar_combined_risk[start_idx[0], start_idx[1]]):
                path = None
            else:
                get_temp_dest_value = 0
                if temp_dest is not None:
                    temp_start_point = (x_edges[start_idx[0]], y_edges[start_idx[1]])
                    get_temp_dest_value = cvar_combined_risk[
                        np.digitize(temp_dest[0], x_edges) - 1, np.digitize(temp_dest[1], y_edges) - 1
                    ]
            trigger_replan = False

            if path is None:
                trigger_replan = True
            else:
                LOOKAHEAD_STEPS = 5
                RISK_THRESHOLD = 0.8
                risks_ahead = [cvar_combined_risk[cell] for i, cell in enumerate(path)
                    if current_target_index <= i < current_target_index + LOOKAHEAD_STEPS
                    and not np.isnan(cvar_combined_risk[cell])]

                if risks_ahead and np.max(risks_ahead) > RISK_THRESHOLD * np.nanmax(cvar_combined_risk):
                    print("Replanning due to high risk ahead.")
                    trigger_replan = True

                # Optional: trigger if vehicle drifts too far from path
                if smoothed_path is not None and current_target_index < len(smoothed_path):
                    dist_to_path = np.linalg.norm(np.array([vehicle_x, vehicle_y]) - smoothed_path[current_target_index])
                    if dist_to_path > 2.0:
                        trigger_replan = True
            print(f"Trigger Replan: {trigger_replan}, Temp Dest Value: {get_temp_dest_value}")
            if trigger_replan:
                    valid_indices = np.argwhere(~np.isnan(cvar_combined_risk))
                    if valid_indices.size > 0:
                        candidate_centers = np.column_stack((x_mid[valid_indices[:, 0]], y_mid[valid_indices[:, 1]]))
                        candidate_distances = np.linalg.norm(candidate_centers - destination_point, axis=1)
                        best_candidate = valid_indices[np.argmin(candidate_distances)]
                        dest_idx = tuple(best_candidate)
                        path = a_star_search(cvar_combined_risk, start_idx, dest_idx)
                        temp_dest = (x_edges[dest_idx[0]], y_edges[dest_idx[1]])
                        update_path = True
                        current_target_index = 0
                    else:
                        path = None

            # ---------------------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------------------
            colors = [(0.5, 0.5, 0.5), (1, 1, 0), (1, 0, 0)]
            cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors)
            ax.clear()
            c = ax.pcolormesh(Y, X, cvar_combined_risk.T, shading='auto', cmap=cmap, alpha=0.7)
            if colorbar is None:
                colorbar = fig.colorbar(c, ax=ax, label='Risk Value (0=zero risk, 1=risky)')
            else:
                colorbar.update_normal(c)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Risk Visualization with A* Path and PID Control')

            # Use backup path if no new one is computed.
            if not update_path and temp_path is not None:
                path = temp_path.copy()
                for j in range(len(path)):
                    path[j] = (np.digitize(path[j][0], x_edges) - 1, np.digitize(path[j][1], y_edges) - 1)

            smoothed_path = None
            if path:
                temp_path = path.copy()
                for i in range(len(temp_path)):
                    temp_path[i] = (x_edges[temp_path[i][0]], y_edges[temp_path[i][1]])
                raw_path = np.array([[x_mid[cell[0]], y_mid[cell[1]]] for cell in path])
                smoothed_path = smooth_path(raw_path, window_size=5)

                # PID Control
                if current_target_index < len(smoothed_path):
                    target_point = smoothed_path[current_target_index]
                    distance_to_target = np.linalg.norm(np.array(target_point) - np.array([vehicle_x, vehicle_y]))
                    if distance_to_target < 2:
                        current_target_index += 2
                        current_target_index = min(current_target_index, len(smoothed_path) - 1)
                        target_point = smoothed_path[current_target_index]
                    desired_heading = math.atan2(target_point[1] - vehicle_y, target_point[0] - vehicle_x)
                    current_heading = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    heading_error = (desired_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
                    steering = steering_pid.compute(heading_error)
                    steering = max(min(steering, 1), -1)
                    dx = target_point[0] - vehicle_x
                    dy = target_point[1] - vehicle_y
                    forward_error = dx * math.cos(current_heading) + dy * math.sin(current_heading)
                    throttle_value = forward_pid.compute(forward_error)
                    throttle = 0 if abs(steering) > 0.75 else throttle_value * ((0.75 - abs(steering)) / 0.75)
                    lidar_test.client.setCarControls(airsim.CarControls(throttle=throttle, steering=steering), lidar_test.vehicleName)
            else:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0.0275, steering=0), lidar_test.vehicleName)

            # --- NEW OPENCV DISPLAY ---
            heading_vector = (rotation_matrix[1, 0], rotation_matrix[0, 0])
            display_risk_map_cv2(
                risk_grid=cvar_combined_risk.T,
                smoothed_path=smoothed_path if path else None,
                vehicle_pos=(vehicle_y, vehicle_x),
                start=(start_idx[1], start_idx[0]),
                dest=(dest_idx[1], dest_idx[0]),
                heading_vector=heading_vector
            )


            distance_last = np.linalg.norm(destination_point - np.array([vehicle_x, vehicle_y]))
            if distance_last < 0.75:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                break

            ax.scatter(vehicle_y, vehicle_x, color="green", label="Start", zorder=5)
            ax.scatter(destination_point[1], destination_point[0], color="red", label="Destination", zorder=5)
            ax.legend()
            plt.draw()
            plt.pause(0.01)

    finally:
        cv2.destroyAllWindows()
