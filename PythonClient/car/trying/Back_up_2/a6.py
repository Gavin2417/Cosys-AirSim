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

import casadi as ca
import numpy as np

class NMPCController:
    def __init__(self, horizon=10, dt=0.1, wheelbase=0.5, V_max=5.0, delta_max=np.deg2rad(25)):
        self.N = horizon
        self.dt = dt
        self.L = wheelbase
        self.V_max = V_max
        self.delta_max = delta_max
        
        # Weights for cost function
        self.Q_x = 1.0
        self.Q_y = 1.0
        self.Q_psi = 1.0
        self.R_v = 0.1
        self.R_delta = 0.1

        # --- symbols ---
        x, y, psi = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('psi')
        states = ca.vertcat(x, y, psi); n_states = states.size1()

        v, delta = ca.SX.sym('v'), ca.SX.sym('delta')
        controls = ca.vertcat(v, delta); n_controls = controls.size1()

        rhs = ca.vertcat(v*ca.cos(psi), v*ca.sin(psi), v/self.L * ca.tan(delta))
        f = ca.Function('f', [states, controls], [rhs])

        U = ca.SX.sym('U', n_controls, self.N)
        X = ca.SX.sym('X', n_states, self.N+1)

        # NEW: dt per shooting interval as parameter
        DT = ca.SX.sym('DT', self.N)

        # Parameters: x0 + (xref,yref)*N + DT*N
        P = ca.SX.sym('P', n_states + 2*self.N + self.N)

        obj = 0
        g = []
        g.append(X[:,0] - P[0:n_states])

        W_first = 5.0
        for k in range(self.N):
            ref_x = P[n_states + 2*k]
            ref_y = P[n_states + 2*k + 1]
            st = X[:,k]
            con = U[:,k]

            # w = W_first - k*(W_first/self.N)
            w = 0.5 + 1.5 * k/(self.N-1)
            obj += w*((st[0]-ref_x)**2 + (st[1]-ref_y)**2 \
                      + (st[2]-ca.atan2(ref_y-st[1], ref_x-st[0]))**2)
            obj += 1e-3 * dt * (con[0]/self.V_max)**2
            obj += 1e-3 * dt * (con[1]/self.delta_max)**2
            if k > 0:
                du = U[:,k] - U[:,k-1]
                obj += 0.05*ca.sumsqr(du)
            # dynamics with variable dt
            dt_k = P[n_states + 2*self.N + k]  # <- pull from P
            st_next = X[:,k+1]
            f_val = f(st, con)
            g.append(st_next - (st + dt_k * f_val))
        # ----- Terminal (final state) cost -----
        xT = X[:, self.N]

        ref_xT = P[n_states + 2*(self.N-1)]
        ref_yT = P[n_states + 2*(self.N-1) + 1]

        # quick heading ref from last segment (or pass psi_ref as a parameter)
        psi_ref_T = ca.atan2(ref_yT - X[1, self.N-1], ref_xT - X[0, self.N-1])

        Qf = ca.diag(ca.SX([20, 20, 5]))   # tune these
        obj += ca.mtimes([(xT - ca.vertcat(ref_xT, ref_yT, psi_ref_T)).T,
                        Qf,
                        (xT - ca.vertcat(ref_xT, ref_yT, psi_ref_T))])
        g = ca.vertcat(*g)
        OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        nlp = {'f': obj, 'x': OPT, 'g': g, 'p': P}
        opts = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # bounds
        lbx, ubx = [], []
        for _ in range(self.N+1):
            lbx += [-ca.inf, -ca.inf, -ca.inf]
            ubx += [ ca.inf,  ca.inf,  ca.inf]
        for _ in range(self.N):
            lbx += [0.02, -self.delta_max]
            ubx += [self.V_max, self.delta_max]

        self.lbx, self.ubx = lbx, ubx
        self.lbg = [0]*g.size1()
        self.ubg = [0]*g.size1()
        self.n_states = n_states
        self.P = P

    def solve(self, x0, ref_traj, dt_seq):
        assert ref_traj.shape[0] >= self.N
        assert len(dt_seq) == self.N

        p = np.concatenate([x0,
                            ref_traj[:self.N].reshape(-1),
                            np.array(dt_seq)])
        x_init = np.tile(x0, (self.N+1, 1))
        u_init = np.zeros((self.N, 2))
        init_guess = np.concatenate([x_init.flatten(), u_init.flatten()])

        sol = self.solver(x0=init_guess,
                          lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=p)
        u_opt = sol['x'][-2*self.N:].full().reshape(self.N, 2)
        return u_opt[0]
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

def a_star_search(cvar_combined_risk, start_idx, dest_idx):
    rows, cols = cvar_combined_risk.shape
    max_risk = np.nanmax(cvar_combined_risk)
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
            if is_valid(nr, nc, cvar_combined_risk) and is_unblocked(cvar_combined_risk, nr, nc, threshold):
                tentative_g = g_scores[current] + cvar_combined_risk[nr, nc]
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

def fade_with_distance_transform(cvar_combined_risk, high_threshold=0.4, fade_scale=4.0, sigma=5.0):
    grid_max = np.nanmax(cvar_combined_risk)
    threshold_val = high_threshold * grid_max
    high_mask = cvar_combined_risk > threshold_val
    dist_map = distance_transform_edt(~high_mask)
    fade_risk = fade_scale * np.exp(-dist_map / sigma)
    return np.maximum(cvar_combined_risk, fade_risk)

if __name__ == "__main__":
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')
    grid_map_ground = GridMap(resolution=0.1)

    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)
    fig, ax = plt.subplots()
    plt.ion()

    path = None
    temp_dest = None
    temp_path = None
    smoothed_path = None
    current_target_index = 0

    horizon =20
    wheelbase = 0.25
    V_max = 0.4
    delta_max = np.deg2rad(25)

    # initial controller (dt placeholder)
    nmpc = NMPCController(horizon=horizon, dt=0.1,
                          wheelbase=wheelbase, V_max=V_max, delta_max=delta_max)
    ctr = airsim.CarControls()

    # Grid setup:
    grid_resolution = 0.1
    margin = 4
    pos, _ = lidar_test.get_vehicle_pose()
    start_point = pos[:2]
    destination_point = np.array([17, -7])
    min_x = min(start_point[0], destination_point[0]) - margin
    max_x = max(start_point[0], destination_point[0]) + margin
    min_y = min(start_point[1], destination_point[1]) - margin
    max_y = max(start_point[1], destination_point[1]) + margin

    x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_mid, y_mid)

    custom_cmap = LinearSegmentedColormap.from_list(
        "gray_yellow_red", [(0.5,0.5,0.5),(1,1,0),(1,0,0)], N=10
    )
    colorbar = None
    prev_time = time.time()
    try:
        while True:
            
            # --- 1) Get & segment LIDAR, update ground grid ---
            pc, ts = lidar_test.get_data(gpulidar=True)
            if pc is None:
                continue

            points = np.array(pc[:,:3])
            points = points[np.linalg.norm(points,axis=1) > 0.6]
            pos, R = lidar_test.get_vehicle_pose()
            world = lidar_test.transform_to_world(points, pos, R)
            world[:,2] = -world[:,2]
            labels = seg.segment(world)

            for p,lab in zip(world, labels):
                grid_map_ground.add_point(p[0], p[1], p[2], lab)

            ground_pts = grid_map_ground.get_label_estimate()

            vehicle_x, vehicle_y = pos[0], pos[1]

            # --- 2) Build CVaR risk grid ---
            gx, gy, gz = ground_pts[:,0], ground_pts[:,1], ground_pts[:,2]
            risk_grid, _, _, _ = binned_statistic_2d(
                gx, gy, gz, statistic='mean', bins=[x_edges, y_edges]
            )
            risk_grid = interpolate_in_radius(risk_grid, 1.5)
            risk_grid = compute_cvar_cellwise(risk_grid, alpha=0.7, radius=4.0)
            risk_grid = np.nan_to_num(risk_grid, nan=1.0)

            # mask far-away
            dist = np.hypot(X - vehicle_x, Y - vehicle_y)
            risk_grid[dist.T > 10.0] = np.nan

            # indices
            start_idx = (np.digitize(vehicle_x, x_edges)-1,
                         np.digitize(vehicle_y, y_edges)-1)
            dest_idx  = (np.clip(np.digitize(destination_point[0], x_edges)-1, 0, len(x_mid)-1),
                         np.clip(np.digitize(destination_point[1], y_edges)-1, 0, len(y_mid)-1))

            # --- 3) Decide if we need to (re)plan A* ---
            update_path = False
            if np.isnan(risk_grid[start_idx]):
                path = None
            trigger = (path is None)
            if not trigger:
                # convert path (list of ints) → world coords
                coords = np.array([[x_mid[i], y_mid[j]] for i,j in path])
                dists  = np.linalg.norm(coords - np.array([vehicle_x, vehicle_y]), axis=1)
                closest = np.argmin(dists)

                lookahead = path[closest:closest+8]
                risks_ahead = [risk_grid[c] for c in lookahead if not np.isnan(risk_grid[c])]

                if not path or (risks_ahead and max(risks_ahead) > 0.6 * np.nanmax(risk_grid)):
                    trigger = True
                if smoothed_path is not None and current_target_index < len(smoothed_path):
                    d2p = np.linalg.norm(np.array([vehicle_x,vehicle_y]) - smoothed_path[current_target_index])
                    if d2p > 1.5:
                        trigger = True

            if trigger:
                valid = np.argwhere(~np.isnan(risk_grid))
                if valid.size:
                    centers = np.column_stack((x_mid[valid[:,0]], y_mid[valid[:,1]]))
                    d2dest = np.linalg.norm(centers - destination_point, axis=1)
                    best   = valid[np.argmin(d2dest)]
                    dest_idx = (best[0], best[1])
                    path = a_star_search(risk_grid, start_idx, dest_idx)
                    temp_dest = (x_edges[dest_idx[0]], y_edges[dest_idx[1]])
                    update_path = True
                    current_target_index = 0
                else:
                    path = None

            # --- 4) Visualization & NMPC on smoothed path ---
            ax.clear()
            c = ax.pcolormesh(Y, X, risk_grid.T, shading='auto',
                              cmap=custom_cmap, alpha=0.7)
            if colorbar is None:
                colorbar = fig.colorbar(c, ax=ax, label='Risk')
            else:
                colorbar.update_normal(c)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title('Risk + A* + NMPC')
            now = time.time()
            prev_time = now
            if path:
                # raw world coords from integer cells
                raw = np.array([[x_mid[i], y_mid[j]] for i,j in path])
                smoothed_path = smooth_path(raw, window_size=5)
                ax.plot(smoothed_path[:,1], smoothed_path[:,0],
                        linewidth=2, label='Smoothed A*')


                d2sp = np.linalg.norm(smoothed_path - pos[:2], axis=1)
                i0 = int(np.argmin(d2sp))

                # get N horizon ref points
                ref_traj = smoothed_path[i0+1:i0+nmpc.N]
                if len(ref_traj) < nmpc.N:
                    ref_traj = np.pad(ref_traj,
                                      ((0, nmpc.N-len(ref_traj)), (0,0)),
                                      mode='edge')

                ax.plot(ref_traj[:,1], ref_traj[:,0], 'bo-', markersize=4, label='NMPC Ref')

                # current yaw
                psi = math.atan2(R[1,0], R[0,0])
                x0  = [pos[0], pos[1], psi]

                # either big turn or NMPC
                next_wp = ref_traj[1] if len(ref_traj)>1 else ref_traj[0]
                dx, dy = next_wp - pos[:2]
                desired = math.atan2(dy, dx)
                err = math.atan2(math.sin(desired-psi), math.cos(desired-psi))
                big_th = np.deg2rad(20)

                # nmpc = NMPCController(horizon=horizon, dt=dt_loop,
                #                   wheelbase=wheelbase, V_max=V_max, delta_max=delta_max)
                dt_loop = max(min(now - prev_time, 0.2), 0.1)
                dt_seq = np.full(nmpc.N, dt_loop)   # or clamp: np.clip(dt_loop, 0.05, 0.2)
                
                v_cmd, d_cmd = nmpc.solve(x0, ref_traj, dt_seq)
                print(f"[DEBUG] v_cmd={v_cmd:.3f}, d_cmd={d_cmd:.3f}")
                print(f"[DEBUG] dt_seq ={dt_seq}")
                stats = nmpc.solver.stats()
                print(stats['return_status'], stats['iter_count'], stats.get('objective', None))
                scale = 1.0 - 0.8 * min(abs(err)/big_th, 1.0)
                ctr.throttle = float(np.clip(v_cmd*scale/nmpc.V_max, 0, 1))
                ctr.steering = float(np.clip(d_cmd/nmpc.delta_max, -1, 1))
 
                # ---- after you build ref_traj ----
                first_wp = ref_traj[0]
                goal_heading = math.atan2(first_wp[1] - pos[1], first_wp[0] - pos[0])
                diff_goal_psi = goal_heading - psi
                err = math.atan2(math.sin(diff_goal_psi), math.cos(diff_goal_psi))
                big_th = np.deg2rad(20)   # keep your value

                if abs(err) > big_th:
                    # Face the first point only
                    ctr.steering = float(np.clip(err / big_th, -1, 1))
                    ctr.throttle = 0.0   # small crawl so you still turn (0 if you truly want to stop)
             
                                    
                
                    
                    

                lidar_test.client.setCarControls(ctr)
            else:
                lidar_test.client.setCarControls(
                    airsim.CarControls(throttle=0.0275, steering=0), lidar_test.vehicleName
                )
            # scatter start & dest
            ax.scatter(vehicle_y, vehicle_x, c='green', s=50, label='You')
            ax.scatter(destination_point[1], destination_point[0],
                       c='red', s=50, label='Goal')
            ax.legend()
            plt.draw()
            plt.pause(0.1)

            # arrival check
            if np.hypot(destination_point[0]-vehicle_x,
                        destination_point[1]-vehicle_y) < 0.75:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0))
                break

    finally:
        plt.ioff()
        plt.show()
        plt.close()