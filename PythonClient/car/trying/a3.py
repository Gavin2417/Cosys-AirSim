# import setup_path
import os, math, time, heapq
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, norm
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure, distance_transform_edt
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
import numpy.ma as ma
import cosysairsim as airsim
import casadi as ca

# from linefit import ground_seg
from function5 import calculate_combined_risks, compute_cvar_cellwise
from scipy.ndimage import generic_filter
base = os.path.dirname(__file__)        
project_root = base
print(project_root)
rand_dir = os.path.join(project_root, "rand")
os.chdir(rand_dir)
from predict1 import RandlaGroundSegmentor


def interpolate_in_radius(grid, radius):
    """
    Vectorized interpolation using cKDTree: fills NaNs in a grid based on nearby valid cells.
    """
    valid_mask = ~np.isnan(grid)
    if np.sum(valid_mask) == 0:
        return grid  # Nothing to interpolate from

    # Grid coordinates
    X, Y = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
    coords = np.stack([X[valid_mask], Y[valid_mask]], axis=1)
    values = grid[valid_mask]

    nan_mask = np.isnan(grid)
    nan_coords = np.stack([X[nan_mask], Y[nan_mask]], axis=1)

    # KDTree on valid points
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(nan_coords, radius)

    for idx, neighbors in enumerate(neighbors_list):
        if neighbors:
            weights = 1.0 / (np.linalg.norm(coords[neighbors] - nan_coords[idx], axis=1) + 1e-6)
            grid[nan_coords[idx][0], nan_coords[idx][1]] = np.sum(weights * values[neighbors]) / np.sum(weights)

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
        self.client = airsim.CarClient(ip="100.123.124.47")
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

    def add_point(self, x, y, z, label):
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = [z, 1]
        else:
            self.grid[cell][0] += label
            self.grid[cell][1] += 1

    def get_label_estimate(self):
        estimates = []
        for (gx, gy), (z_sum, count) in self.grid.items():
            # get rhe mean label
            mean_label = np.ceil(z_sum/count)
            estimates.append([gx * self.resolution, gy * self.resolution, mean_label])
        return np.array(estimates)



class NMPCController:
    def __init__(self, horizon=10, dt=0.1, wheelbase=0.5, V_max=5.0, delta_max=np.deg2rad(25)):
        self.N = horizon
        self.dt = dt
        self.L = wheelbase
        self.V_max = V_max
        self.delta_max = delta_max
        
        # Weights for cost function
        self.Q_x = 3.0
        self.Q_y = 3.0
        self.Q_psi = 2.0
        self.R_v = 1e-4
        self.R_delta = 2.0

        # Define state and control symbols
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        states = ca.vertcat(x, y, psi)
        n_states = states.size()[0]

        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta)
        n_controls = controls.size()[0]

        # Kinematic bicycle model
        rhs = ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v/self.L * ca.tan(delta)
        )
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables
        U = ca.SX.sym('U', n_controls, self.N)
        X = ca.SX.sym('X', n_states, self.N + 1)

        # Parameter vector: initial state + reference trajectory (x,y) for each horizon step
        P = ca.SX.sym('P', n_states + 2*self.N)

        # Objective and constraints lists
        obj = 0
        g = []

        # Initial condition constraint
        g.append(X[:, 0] - P[0:n_states])

        # Build the MPC optimization
        for k in range(self.N):
            # Reference for current step
            ref_x = P[n_states + 2*k]
            ref_y = P[n_states + 2*k + 1]
            st = X[:, k]
            con = U[:, k]

            # Cost: tracking + control effort
            obj += self.Q_x * (st[0] - ref_x)**2
            obj += self.Q_y * (st[1] - ref_y)**2
            obj += self.Q_psi * (st[2] - ca.atan2(ref_y - st[1], ref_x - st[0]))**2
            obj += self.R_v * (con[0]/self.V_max)**2
            obj += self.R_delta * (con[1]/self.delta_max)**2

            # System dynamics (Euler integration)
            st_next = X[:, k+1]
            f_value = f(st, con)
            st_next_euler = st + self.dt * f_value
            g.append(st_next - st_next_euler)

        # Concatenate constraints
        g = ca.vertcat(*g)

        # Decision variables vector
        OPT_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # NLP problem
        nlp_dict = {'f': obj, 'x': OPT_vars, 'g': g, 'p': P}
        opts = {'ipopt.max_iter': 100, 'ipopt.print_level': 5, 'print_time': 1}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

        # Bounds for optimization variables
        lbx = []
        ubx = []
        # State bounds
        for _ in range(self.N + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf]
            ubx += [ ca.inf,  ca.inf,  ca.inf]
        # Control bounds
        for _ in range(self.N):
            lbx += [0.0, -self.delta_max]
            ubx += [ self.V_max,  self.delta_max]
        self.lbx = lbx
        self.ubx = ubx

        # Bounds for constraints (equalities)
        self.lbg = [0] * g.size()[0]
        self.ubg = [0] * g.size()[0]

    def solve(self, x0, ref_traj):
        """
        Solve the NMPC problem.
        :param x0: Current state [x, y, psi]
        :param ref_traj: Array of shape (N, 2) with reference (x, y) points
        :return: optimal control [v, delta] for the first step
        """
        # Build parameter vector
        p = list(x0) + ref_traj.flatten().tolist()

        # Initial guess for states and controls
        x_init = np.tile(x0, (self.N+1, 1))
        u_init = np.tile([0.2, 0.0], (self.N, 1))
        init_guess = np.concatenate((x_init.flatten(), u_init.flatten()))

        # Solve
        sol = self.solver(
            x0=init_guess,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )

        # Extract control sequence
        u_opt = sol['x'][-2*self.N:].full().reshape(self.N, 2)
        return u_opt[0]  # first control action


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

    # Initialize ground segmentation.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
    seg = RandlaGroundSegmentor(
    device=None,
    subsample_grid=0.1
    )

    fig, ax = plt.subplots()
    plt.ion()
    colorbar = None
    path = None
    temp_dest = None
    nmpc = NMPCController(
        horizon=2,      # how many steps to look ahead
        dt=0.1,          # same as your main loop
        wheelbase=0.5,   # Husky wheelbase [m]
        V_max=1.0,       # tune to your platform
        delta_max=np.deg2rad(30)
    )
    current_target_index = 0

    # Define grid boundaries based on vehicle and destination.
    grid_resolution = 0.1
    margin = 4
    position, rotation_matrix = lidar_test.get_vehicle_pose()
    start_point = np.array([position[0], position[1]])
    destination_point = np.array([17, -7])
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
    smoothed_path = None
    colors = [(0.5, 0.5, 0.5),  # gray
          (1,   1,   0),    # yellow
          (1,   0,   0)]    # red
    custom_cmap = LinearSegmentedColormap.from_list("gray_yellow_red", colors, N=10)

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
            labels = seg.segment(points_world)

            # Populate grid maps based on segmentation.
            for i, point in enumerate(points_world):
                x, y, z = point
                point_label = labels[i]
                grid_map_ground.add_point(x, y, z, point_label)

            ground_points = grid_map_ground.get_label_estimate()
            vehicle_x, vehicle_y = position[0], position[1]
            center = np.array([vehicle_x, vehicle_y])
        

            # # Build the ground grid using vectorized binning.
            ground_x_vals = ground_points[:, 0]
            ground_y_vals = ground_points[:, 1]
            ground_z_vals = ground_points[:, 2]
            cvar_combined_risk, _, _, _ = binned_statistic_2d(
                ground_x_vals, ground_y_vals, ground_z_vals, statistic='mean', bins=[x_edges, y_edges]
            )



            # max_risk = np.nanmax(total_risk_grid)
            # threshold = 0.20 * max_risk
            # mask = total_risk_grid > threshold
            # total_risk_grid[mask] = np.exp(total_risk_grid[mask])
            cvar_combined_risk = interpolate_in_radius(cvar_combined_risk, 1.5)
            # masked_total_risk_grid = ma.masked_invalid(total_risk_grid)
            cvar_combined_risk = compute_cvar_cellwise(cvar_combined_risk, alpha=0.7, radius=4.0)
            cvar_combined_risk = np.nan_to_num(cvar_combined_risk, nan=1)

            # # Mask cells far from the vehicle.
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
            trigger_replan = path is None
            if not trigger_replan:
                vehicle_pos = np.array([vehicle_x, vehicle_y])
                path_coords = np.array([[x_mid[cell[0]], y_mid[cell[1]]] for cell in path])
                distances = np.linalg.norm(path_coords - vehicle_pos, axis=1)
                closest_index = np.argmin(distances)
                LOOKAHEAD_STEPS = 8
                RISK_THRESHOLD = 0.60
                lookahead_cells = path[closest_index : closest_index + LOOKAHEAD_STEPS]
                risks_ahead = [
                    cvar_combined_risk[cell]
                    for cell in lookahead_cells
                    if not np.isnan(cvar_combined_risk[cell])
                ]

                # Trigger replan if needed
                trigger_replan = False

                if not path or (risks_ahead and np.max(risks_ahead) > RISK_THRESHOLD * np.nanmax(cvar_combined_risk)):
                    trigger_replan = True
                # Optional: trigger if vehicle drifts too far from path
                if smoothed_path is not None and current_target_index < len(smoothed_path):
                    dist_to_path = np.linalg.norm(np.array([vehicle_x, vehicle_y]) - smoothed_path[current_target_index])
                    if dist_to_path > 1.5:
                        trigger_replan = True

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

            if path:
                temp_path = path.copy()
                for i in range(len(temp_path)):
                    temp_path[i] = (x_edges[temp_path[i][0]], y_edges[temp_path[i][1]])
                raw_path = np.array([[x_mid[cell[0]], y_mid[cell[1]]] for cell in path])
                smoothed_path = smooth_path(raw_path, window_size=5)
                ax.plot(smoothed_path[:, 1], smoothed_path[:, 0],
                        color="blue", linewidth=2, label="Smoothed A* Path")

                # -----------------------------------------------------------------
                # PID Control: Follow the computed (smoothed) A* path.
                # -----------------------------------------------------------------
                if current_target_index < len(smoothed_path):
                    # find index of closest point on smoothed_path
                    vehicle_pos = np.array([vehicle_x, vehicle_y])
                    dists = np.linalg.norm(smoothed_path - vehicle_pos, axis=1)
                    # find index of closest point on smoothed_path
                    i_closest = int(np.argmin(np.linalg.norm(smoothed_path - vehicle_pos, axis=1)))
                
                    # start one step *beyond* the closest, so MPC actually has to move
                    start_idx = min(i_closest + 1, len(smoothed_path)-1)
                    ref_pts = smoothed_path.tolist()
                    ref_list = ref_pts[start_idx : start_idx + nmpc.N]
                    # pad with final point if too short
                    if len(ref_list) < nmpc.N:
                        ref_list += [ref_pts[-1]] * (nmpc.N - len(ref_list))
                    ref_traj = np.array(ref_list)

                    # get current yaw
                    current_heading = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
                    x0 = [vehicle_x, vehicle_y, current_heading]

                    # big‐turn fallback
                    next_wp = ref_traj[0]


                    desired_psi = math.atan2(next_wp[1]-vehicle_y, next_wp[0]-vehicle_x)
                    raw_error = desired_psi - current_heading
                    heading_error = math.atan2(math.sin(raw_error), math.cos(raw_error))
                    big_turn_thresh = np.deg2rad(40)
                    ctr = airsim.CarControls()
                    
            
                    v_cmd, delta_cmd = nmpc.solve(x0, ref_traj)
            
                    # throttle [0 … 0.5]
                    throttle_ratio = np.clip(v_cmd / nmpc.V_max, 0.0, 1.0)
                    ctr.throttle = float(throttle_ratio)

                    # steering [–1 … 1]
                    ctr.steering = float(np.clip(delta_cmd / nmpc.delta_max, -1.0, 1.0))
                    # print(ctr.steering, " ", ctr. throttle)
                    # if abs(heading_error) > big_turn_thresh:
                    #     ctr.throttle = 0.0
     

                    lidar_test.client.setCarControls(ctr)
            else:
                lidar_test.client.setCarControls(
                    airsim.CarControls(throttle=0.0275, steering=0), lidar_test.vehicleName
                )

            distance_last = np.linalg.norm(destination_point - np.array([vehicle_x, vehicle_y]))
            if distance_last < 0.75:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0), lidar_test.vehicleName)
                break

            ax.scatter(vehicle_y, vehicle_x, color="green", label="Start", zorder=5)
            ax.scatter(destination_point[1], destination_point[0], color="red", label="Destination", zorder=5)
            ax.legend()
            plt.draw()
            plt.pause(0.1)

    finally:
        plt.ioff()
        plt.show()
        plt.close()
