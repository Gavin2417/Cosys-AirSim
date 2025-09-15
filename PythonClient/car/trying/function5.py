import os, math, time, heapq, json, argparse
import numpy as np
import open3d as o3d
import numpy.ma as ma
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.stats import norm
import cosysairsim as airsim
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.4, max_slope_degrees=30.0, radius=0.3):
    """
    Vectorized step & slope risk over an 8‐neighbor window.
    The non_nan_indices argument is kept for compatibility but not used.
    """
    # Precompute constants
    max_slope_rad = np.deg2rad(max_slope_degrees)
    diag_dist = np.sqrt(radius**2 + radius**2)

    # Define the 8 neighbor shifts
    shifts = [
        ( 0,  1), ( 0, -1),
        ( 1,  0), (-1,  0),
        ( 1,  1), ( 1, -1),
        (-1,  1), (-1, -1),
    ]

    # Compute absolute height differences for each shift
    diffs = []
    for dx, dy in shifts:
        shifted = np.roll(np.roll(Z_grid, dx, axis=0), dy, axis=1)
        diffs.append(np.abs(shifted - Z_grid))

    all_diffs = np.stack(diffs, axis=0)

    # Mask out differences where either cell was NaN
    nan_mask = np.isnan(Z_grid)
    all_diffs[:, nan_mask] = 0

    # Maximum neighbor difference per cell
    max_diff = np.max(all_diffs, axis=0)

    # Step risk: normalized and capped
    step_risk = np.minimum(max_diff / max_height_diff, 1.0)

    # Slope risk: arctan of gradient over diagonal distance, normalized and capped
    slope_risk = np.minimum((np.arctan(max_diff / diag_dist) / max_slope_rad), 1.0)

    # Restore NaNs where input was NaN
    step_risk[nan_mask] = np.nan
    slope_risk[nan_mask] = np.nan

    return step_risk, slope_risk


def compute_cvar_cellwise(risk_grid, alpha=0.2, radius=5.0):
    """
    For each valid cell, collect all risk values within `radius` (in grid cells),
    compute the α‑quantile (VaR) of that local sample, then average all local values
    ≥ VaR to get the empirical CVaR.
    """
    rows, cols = risk_grid.shape
    cvar = np.full_like(risk_grid, np.nan)

    # 1) Build a KD‑tree of all valid‐risk cell coords
    valid_mask = ~np.isnan(risk_grid)
    coords = np.column_stack(np.where(valid_mask))
    values = risk_grid[valid_mask]
    tree = cKDTree(coords)

    # 2) For each valid cell, query its neighborhood
    for idx, (r, c) in enumerate(coords):
        neigh_idx = tree.query_ball_point((r, c), radius)
        local_vals = values[neigh_idx]
        if local_vals.size == 0:
            continue

        clean = np.ma.compressed(local_vals)    # gives a 1‑D ndarray of just the unmasked values
        if clean.size == 0:
            continue
        var = np.quantile(clean, alpha)
        # 4) Average the tail ≥ VaR
        tail = local_vals[local_vals >= var]
        cvar[r, c] = tail.mean() if tail.size else var

    return cvar

### CLASS
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

class AStarPlanner:
    def __init__(self,
                 grid: np.ndarray,
                 risk_factor: float = 0.8,
                 surround_weight: float = 1.0,
                 surround_sigma: float = 3.0):
        self.grid = grid.copy()
        max_risk = np.nanmax(self.grid) if not np.isnan(np.nanmax(self.grid)) else 1.0
        self.threshold = risk_factor * max_risk
        self.rows, self.cols = grid.shape

        # 1) build high‑risk mask
        high_mask = (self.grid >= self.threshold) | np.isnan(self.grid)

        # 2) distance from “safe” regions
        dist = distance_transform_edt(~high_mask)

        # 3) fade cost: big near high-mask, decays with sigma
        self.proximity_cost = surround_weight * np.exp(-dist / surround_sigma)

        # 4) final cost map: sum of raw risk + proximity penalty
        #    (nan→very large to keep blocked cells blocked)
        self.cost_map = np.where(np.isnan(self.grid),
                                 np.inf,
                                 self.grid + self.proximity_cost)

    def _heuristic(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def _reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]

    def plan(self, start, goal, MAX_RTSK_VALUE=50, max_expansions=20000):
        # check validity
        for pt in (start, goal):
            r, c = pt
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if self.cost_map[r, c] == np.inf:
                return None

        # Compute risk threshold
        # max_risk = np.max(self.cost_map[np.isfinite(self.cost_map
        risk_threshold = self.threshold

        open_set = []
        g_score = {start: 0.0}
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}
        closed = set() 

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

        expansions = 0
        best_so_far = None
        best_f = float('inf')

        while open_set:
            f, current = heapq.heappop(open_set)
            if current in closed:   # <--- skip if already expanded
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if expansions >= max_expansions:
                # give a partial path toward the best node so far
                if best_so_far is not None:
                    return self._reconstruct_path(came_from, best_so_far)
                return None

            expansions += 1
            cg = g_score[current]
            for dr, dc in neighbors:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                cell_cost = self.cost_map[nr, nc]
                if cell_cost == np.inf or cell_cost >= risk_threshold:
                    continue

                # small heuristic: skip tiny improvements to curb thrash
                step_cost = cell_cost * np.hypot(dr, dc)
                tentative = cg + step_cost
                neighbor = (nr, nc)

                if tentative < g_score.get(neighbor, np.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative + self._heuristic(neighbor, goal), neighbor))
        return None

class NMPCController:
    def __init__(self, horizon=10, dt=0.1, wheelbase=0.5,
                 V_max=0.5, delta_max=np.deg2rad(25)):
        self.N        = horizon
        self.dt       = dt
        self.L        = wheelbase
        self.V_max    = V_max
        self.delta_max= delta_max

        # Tuning weights
        self.Q_pose   = np.diag([5, 5, 2])     # x,y,ψ tracking
        self.R_u      = np.diag([0.01, 0.01])  # v,δ effort
        self.R_du     = 0.05                   # smoothness penalty

        # symbols
        x, y, psi = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('psi')
        states  = ca.vertcat(x, y, psi)
        v, dlt  = ca.SX.sym('v'), ca.SX.sym('dlt')
        controls = ca.vertcat(v, dlt)

        # dynamics
        rhs = ca.vertcat(v*ca.cos(psi),
                         v*ca.sin(psi),
                         v/self.L * ca.tan(dlt))
        f   = ca.Function('f', [states, controls], [rhs])

        # decision variables
        X = ca.SX.sym('X', 3, self.N+1)
        U = ca.SX.sym('U', 2, self.N)

        # parameters: [ x0(3), ref_x/ref_y (2*N), dt_seq (N) ]
        P = ca.SX.sym('P', 3 + 2*self.N + self.N)

        g   = []
        obj = 0

        # initial-state
        g.append(X[:,0] - P[0:3])

        for k in range(self.N):
            xr = P[3 + 2*k]
            yr = P[3 + 2*k + 1]
            dt_k = P[3 + 2*self.N + k]

            st = X[:,k]
            uc = U[:,k]

            # tracking cost
            # tracking cost (safe heading)
            des_psi = ca.atan2(yr - st[1], (xr - st[0]) + 1e-8)
            ang_err = ca.atan2(ca.sin(st[2] - des_psi), ca.cos(st[2] - des_psi))
            err = ca.vertcat(st[0] - xr, st[1] - yr, ang_err)
            obj += ca.mtimes([err.T, self.Q_pose, err])

            # control effort
            obj += ca.mtimes([uc.T, self.R_u, uc]) * dt_k

            # smoothness
            if k>0:
                du = U[:,k] - U[:,k-1]
                obj += self.R_du * ca.sumsqr(du)

            # dynamics
            st_next = X[:,k+1]
            fval    = f(st, uc)
            g.append(st_next - (st + dt_k * fval))

        # terminal cost
        errT = X[:,self.N] - ca.vertcat(
            P[3+2*(self.N-1)],
            P[3+2*(self.N-1)+1],
            0
        )
        Qf   = np.diag([10,10,5])
        obj += ca.mtimes([errT.T, Qf, errT])

        # build the NLP
        G   = ca.vertcat(*g)
        OPT = ca.vertcat(ca.reshape(X, -1,1),
                         ca.reshape(U, -1,1))
        nlp = {'f': obj, 'x':OPT, 'g':G, 'p':P}
        opts = {
            # IPOPT itself
            'ipopt.print_level':           0,      # no iteration‐by‐iteration printouts
            'ipopt.sb':                    'yes',  # suppress solver banner
            'ipopt.print_timing_statistics':'no',  # no timing stats
            # CasADi wrapper
            'print_time':                  False,  # don’t print overall timing
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # bounds (X free, U in [0,V_max]×[-δ_max,δ_max])
        nX = 3*(self.N+1)
        self.lbx = [-ca.inf]*nX + [0, -self.delta_max]*self.N
        self.ubx = [ ca.inf]*nX + [self.V_max, self.delta_max]*self.N
        self.lbg = [0]*G.size1()
        self.ubg = [0]*G.size1()

    def solve(self, x0, ref_traj, dt_seq):
        N = self.N
        assert len(dt_seq)==N

        p = np.concatenate([x0, ref_traj[:N].reshape(-1), np.array(dt_seq)])
        x_init = np.tile(x0, (N+1,1))
        u_init = np.zeros((N,2))
        init   = np.concatenate([x_init.flatten(), u_init.flatten()])

        sol = self.solver(x0=init,
                          lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=p)
        U_opt = sol['x'][-2*N:].full().reshape(N,2)
        return U_opt[0]  
### FUNCTIONS

def smooth_path(path, window_size=5):
    path = np.array(path)
    n_points = len(path)
    if n_points < window_size:
        return path
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    sm = [np.mean(path[max(0, i-half):min(n_points, i+half+1)], axis=0)
          for i in range(n_points)]
    return np.array(sm)
def interpolate_in_radius(grid, radius):
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
def filter_points_by_radius(points, center, radius):
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[distances <= radius]
def get_map_setting(sp, dp, margin, grid_resolution):
    # sp = start_point (x,y), dp = destination_point (x,y)
    min_x = min(sp[0], dp[0]) - margin
    max_x = max(sp[0], dp[0]) + margin
    min_y = min(sp[1], dp[1]) - margin
    max_y = max(sp[1], dp[1]) + margin

    # compute the grid edges
    x_edges = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    y_edges = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    x_mid = (x_edges[:-1] + x_edges[1:]) / 2
    y_mid = (y_edges[:-1] + y_edges[1:]) / 2

    return x_edges, y_edges, x_mid, y_mid
def serialize(obj):
    if hasattr(obj, "__dict__"):
        return {k: serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    else:
        # primitive (int, float, bool, str, etc.)
        return obj
def fade_with_distance_transform(risk_grid, high_threshold=0.4, fade_scale=4.0, sigma=5.0):
    grid_max = np.nanmax(risk_grid)
    threshold_val = high_threshold * grid_max
    high_mask = risk_grid > threshold_val
    dist_map = distance_transform_edt(~high_mask)
    fade_risk = fade_scale * np.exp(-dist_map / sigma)
    return np.maximum(risk_grid, fade_risk)