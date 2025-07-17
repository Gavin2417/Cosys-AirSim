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
# from linefit import ground_seg
from function5 import calculate_combined_risks, compute_cvar_cellwise
from scipy.ndimage import generic_filter
base = os.path.dirname(__file__)        
project_root = base
print(project_root)
rand_dir = os.path.join(project_root, "rand")
os.chdir(rand_dir)
from predict1 import RandlaGroundSegmentor
# ─────────────────────────────────────────────────────────────────────────────
# (1) New imports for MPC
import cvxpy as cp
import cvxpy as cp
import numpy as np

class MPCController:
    def __init__(self,
                 N=10, dt=0.1,
                 v_max=1.0,      # m/s
                 omega_max=1.0,  # rad/s
                 Q_pos=1.0, Q_heading=0.1,
                 R_v=0.01, R_omega=0.01,
                 Rd_v=1e-3, Rd_omega=1e-3):
        # horizon & timestep
        self.N = N
        self.dt = dt

        # input limits
        self.v_max     = v_max
        self.omega_max = omega_max

        # weights
        self.Q_pos     = Q_pos
        self.Q_heading = Q_heading
        self.R_v       = R_v
        self.R_omega   = R_omega
        self.Rd_v      = Rd_v
        self.Rd_omega  = Rd_omega

        # dims: state = [x,y,θ], control = [v,ω]
        nx, nu = 3, 2

        # decision vars
        self.x = cp.Variable((nx, N+1))
        self.u = cp.Variable((nu, N))

        # params to update each solve
        self.x0    = cp.Parameter(nx)
        self.x_ref = cp.Parameter((nx, N+1))
        self.A = [cp.Parameter((nx,nx)) for _ in range(N)]
        self.B = [cp.Parameter((nx,nu)) for _ in range(N)]
        self.c = [cp.Parameter(nx)      for _ in range(N)]

        # build cost
        cost = 0
        for k in range(N):
            e_pos = self.x[:2, k] - self.x_ref[:2, k]
            e_th  = self.x[2,   k] - self.x_ref[2, k]
            cost += Q_pos     * cp.sum_squares(e_pos)
            cost += Q_heading * cp.square(e_th)
            cost += R_v     * cp.square(self.u[0,k])
            cost += R_omega * cp.square(self.u[1,k])
            if k > 0:
                cost += Rd_v     * cp.square(self.u[0,k]   - self.u[0,k-1])
                cost += Rd_omega * cp.square(self.u[1,k]   - self.u[1,k-1])
        # terminal pos cost
        eN = self.x[:2, N] - self.x_ref[:2, N]
        cost += Q_pos * cp.sum_squares(eN)

        # constraints
        cons = [ self.x[:,0] == self.x0 ]
        for k in range(N):
            cons += [
                # linearized dyn
                self.x[:,k+1] == self.A[k] @ self.x[:,k]
                                 + self.B[k] @ self.u[:,k]
                                 + self.c[k],
                # input bounds
                cp.abs(self.u[0,k]) <= self.v_max,
                cp.abs(self.u[1,k]) <= self.omega_max,
            ]

        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def solve(self, x_init, x_ref_traj, u_ref_traj=None):
        """
        x_init: (3,) array [x,y,θ]
        x_ref_traj: shape (3, N+1)
        u_ref_traj: shape (2, N) nominal inputs for linearization (optional)
        """
        self.x0.value    = x_init
        self.x_ref.value = x_ref_traj

        if u_ref_traj is None:
            u_ref_traj = np.zeros((2, self.N))

        # build & upload A, B, c at each step
        for k in range(self.N):
            xk = x_ref_traj[:,k]
            uk = u_ref_traj[:,k]
            θk, vk, ωk = xk[2], uk[0], uk[1]

            # Jacobians of f(x,u) = x + [v cosθ, v sinθ, ω]·dt
            A_k = np.eye(3)
            A_k[0,2] = -vk * np.sin(θk) * self.dt
            A_k[1,2] =  vk * np.cos(θk) * self.dt

            B_k = np.zeros((3,2))
            B_k[0,0] = np.cos(θk) * self.dt
            B_k[1,0] = np.sin(θk) * self.dt
            B_k[2,1] = 1.0 * self.dt

            # nominal f
            f_nom = xk + np.array([
                vk*np.cos(θk)*self.dt,
                vk*np.sin(θk)*self.dt,
                ωk*self.dt
            ])

            c_k = f_nom - A_k.dot(xk) - B_k.dot(uk)

            self.A[k].value = A_k
            self.B[k].value = B_k
            self.c[k].value = c_k

        # solve QP
        self.prob.solve(solver=cp.OSQP, warm_start=True)
        # return first control action (v, ω)
        return float(self.u[0,0].value), float(self.u[1,0].value)



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
    # wheel_base for δ = atan2(ω·L, v) and max_steer for normalizing δ
    wheel_base      = 0.5           # [m] your Husky’s track-to-track width in Unreal
    max_steer_angle = np.deg2rad(30) # [rad] your max steering deflection

    mpc = MPCController(
        N=10,
        dt=0.1,
        v_max=0.5,        # max forward speed [m/s]
        omega_max=1.0,    # max yaw rate [rad/s]
        Q_pos=1.0,
        Q_heading=0.1,
        R_v=0.01,
        R_omega=0.01,
        Rd_v=1e-3,
        Rd_omega=1e-3
    )
    # store for later δ computation
    mpc.L = wheel_base
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
    seg = RandlaGroundSegmentor()
    fig, ax = plt.subplots()
    plt.ion()
    colorbar = None
    path = None
    temp_dest = None
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
                                    # -----------------------------------------------------------------
                    # MPC Control: Prepare a reference trajectory over the next N steps
                    # -----------------------------------------------------------------
                    # 1) Build x_ref: shape (4, N+1): [x,y,theta,v] from smoothed_path
                    ref_states = np.zeros((4, mpc.N+1))
                    # fill (x,y,theta) from smoothed_path around current_target_index
                    for k in range(mpc.N+1):
                        idx = min(current_target_index + k, len(smoothed_path)-1)
                        px, py = smoothed_path[idx]
                        # approximate reference heading from path tangent
                        if idx < len(smoothed_path)-1:
                            dx, dy = smoothed_path[idx+1] - smoothed_path[idx]
                            pref_th = math.atan2(dy, dx)
                        else:
                            pref_th = math.atan2(py - smoothed_path[idx-1][1],
                                                px - smoothed_path[idx-1][0])
                        ref_states[0, k] = px
                        ref_states[1, k] = py
                        ref_states[2, k] = pref_th
                        # you can set a desired constant speed, e.g. 1.0 m/s
                        ref_states[3, k] = 1.0

                    # 2) Current state x_init = [x, y, theta, v]
                    #    Get current speed from AirSim if available, else approximate
                    car_state = lidar_test.client.getCarState(lidar_test.vehicleName)
                    speed = car_state.speed
                    # extract current heading from rotation_matrix
                    curr_th = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
                    x_init = np.array([vehicle_x, vehicle_y, curr_th, speed])

                        # -----------------------------------------------------------------
                    # MPC Control (differential‐drive → throttle & steering)
                    # -----------------------------------------------------------------
                    # 1) Build x_ref: shape (3, N+1) = [x; y; θ]
                    x_ref = np.zeros((3, mpc.N+1))
                    for k in range(mpc.N+1):
                        idx = min(current_target_index + k, len(smoothed_path)-1)
                        px, py = smoothed_path[idx]
                        if idx < len(smoothed_path)-1:
                            dx, dy = smoothed_path[idx+1] - smoothed_path[idx]
                            th_ref = math.atan2(dy, dx)
                        else:
                            prev = smoothed_path[idx-1]
                            th_ref = math.atan2(py - prev[1], px - prev[0])
                        x_ref[:, k] = [px, py, th_ref]

                    # 2) Current state [x, y, θ]
                    curr_th = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
                    x_init = np.array([vehicle_x, vehicle_y, curr_th])

                    # 3) Solve MPC → (v_cmd, ω_cmd)
                    v_cmd, omega_cmd = mpc.solve(x_init, x_ref)

                    # 4) Map v_cmd → throttle (assumes v_max is your top speed)
                    throttle = float(np.clip(v_cmd / mpc.v_max, -1.0, 1.0))

                    # 5) Convert ω_cmd → steering δ via bicycle relation δ = atan2(ω·L, v)
                    #    then normalize δ / max_steer
                    if abs(v_cmd) > 1e-3:
                        delta = math.atan2(omega_cmd * mpc.L, v_cmd)
                    else:
                        delta = 0.0
                    steer = float(np.clip(delta / max_steer_angle, -1.0, 1.0))

                    # 6) Send CarControls
                    controls = airsim.CarControls(throttle=throttle*(-1.0), steering=steer)
                    lidar_test.client.setCarControls(controls, lidar_test.vehicleName)


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
