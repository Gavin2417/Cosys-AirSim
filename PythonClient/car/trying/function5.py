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
import numpy as np
from scipy.ndimage import convolve
from numba import njit, prange


def calculate_combined_risks(Z_grid, non_nan_indices, max_height_diff=0.4, max_slope_degrees=30.0, radius=0.3):
    """
    Step & slope risk over an 8‑neighbor window without wrap-around.
    - Keeps outputs exactly same shape as Z_grid
    - Uses per-neighbor spacing (radius for axial, sqrt(2)*radius for diagonals)
    - non_nan_indices kept for compatibility (unused)
    """
    # Precompute constants
    max_slope_rad = np.deg2rad(max_slope_degrees)

    # Define the 8 neighbor shifts and their distances
    shifts = [
        ( 0,  1), ( 0, -1),
        ( 1,  0), (-1,  0),
        ( 1,  1), ( 1, -1),
        (-1,  1), (-1, -1),
    ]

    def neighbor_diff_no_wrap(Z, dx, dy):
        # Slices for the overlapping region between Z and its (dx,dy) shift
        r0 = max(0, dx)
        r1 = Z.shape[0] + min(0, dx)
        c0 = max(0, dy)
        c1 = Z.shape[1] + min(0, dy)
        base  = Z[r0:r1, c0:c1]
        neigh = Z[r0-dx:r1-dx, c0-dy:c1-dy]
        diff_slice = np.abs(base - neigh)

        # zero out where either is NaN
        invalid = ~np.isfinite(base) | ~np.isfinite(neigh)
        diff_slice[invalid] = 0.0

        # place into full-size canvas
        out = np.zeros_like(Z, dtype=float)
        out[r0:r1, c0:c1] = diff_slice
        return out

    # Collect per-neighbor diffs and gradients
    diffs = []
    grads = []
    for dx, dy in shifts:
        d = neighbor_diff_no_wrap(Z_grid, dx, dy)
        diffs.append(d)
        # neighbor spacing per direction
        if abs(dx) + abs(dy) == 1:
            dist = radius
        else:
            dist = radius * np.sqrt(2.0)
        # avoid division by zero if radius is 0
        dist = max(dist, 1e-9)
        grads.append(d / dist)

    all_diffs = np.stack(diffs, axis=0)           # [8,H,W]
    all_grads = np.stack(grads, axis=0)           # [8,H,W]

    # Maximum neighbor difference/gradient per cell
    max_diff = np.max(all_diffs, axis=0)
    max_grad = np.max(all_grads, axis=0)

    # Step risk: normalized and capped
    step_risk = np.minimum(max_diff / max_height_diff, 1.0)

    # Slope risk: arctan of gradient normalized and capped
    slope_risk = np.minimum(np.arctan(max_grad) / max_slope_rad, 1.0)

    # Restore NaNs where input was NaN
    nan_mask = np.isnan(Z_grid)
    step_risk[nan_mask] = np.nan
    slope_risk[nan_mask] = np.nan

    return step_risk, slope_risk



def compute_cvar_cellwise(risk_grid, alpha=0.2, radius=5.0):
    """
    Empirical CVaR over the *upper* alpha tail within a ball of 'radius' (in cells).
    Handles MaskedArray and NaNs safely.
    """
    # normalize input to plain float ndarray with NaNs
    if np.ma.isMaskedArray(risk_grid):
        rg = risk_grid.filled(np.nan).astype(float, copy=False)
    else:
        rg = np.array(risk_grid, dtype=float, copy=False)

    cvar = np.full(rg.shape, np.nan, dtype=float)

    valid_mask = ~np.isnan(rg)
    coords = np.column_stack(np.where(valid_mask))
    if coords.size == 0:
        return cvar

    values = rg[valid_mask]  # plain ndarray of valid numbers
    tree = cKDTree(coords)

    upper_q = 1.0 - alpha
    for (r, c) in coords:
        idxs = tree.query_ball_point((r, c), radius)
        local_vals = values[idxs]
        if local_vals.size == 0:
            continue
        q = np.quantile(local_vals, upper_q)    # no NaNs here
        tail = local_vals[local_vals >= q]
        cvar[r, c] = tail.mean() if tail.size else q

    return cvar

### CLASS
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        # self.client = airsim.CarClient(ip="100.123.124.103")
        self.client = airsim.CarClient(ip="192.168.68.101")
        # print("Connected to client")
        self.client.confirmConnection()
        # print("Confirmed connection")
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
                 surround_sigma: float = 3.0,
                 risk_weight: float = 6.0,
                 risk_power: float = 2.0,
                 prox_weight: float = 1.0):
        self.grid = grid.copy()
        max_risk_raw = np.nanmax(self.grid)
        if not np.isfinite(max_risk_raw):
            max_risk_raw = 1.0
        self.max_risk = max_risk_raw
        self.threshold = risk_factor * self.max_risk
        self.rows, self.cols = grid.shape

        high_mask = (self.grid >= self.threshold) | np.isnan(self.grid)
        dist = distance_transform_edt(~high_mask)
        self.proximity_cost = surround_weight * np.exp(-dist / surround_sigma)

        # risk shaping for fallback (route_through_array) and visualization
        norm = np.where(np.isfinite(self.grid), self.grid / (self.max_risk + 1e-9), np.nan)
        shaped = (norm ** risk_power) * risk_weight
        self.cost_map = np.where(np.isnan(self.grid),
                                 np.inf,
                                 shaped + prox_weight * self.proximity_cost)

        self.risk_weight = risk_weight
        self.risk_power  = risk_power
        self.prox_weight = prox_weight

    def _heuristic(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def _reconstruct_path(self, came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]

    def plan(self, start, goal, MAX_RTSK_VALUE=50, max_expansions=20000):
        for pt in (start, goal):
            r, c = pt
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if not np.isfinite(self.grid[r, c]) or self.grid[r, c] >= self.threshold:
                return None

        risk_threshold = self.threshold

        open_set = []
        g_score = {start: 0.0}
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}
        closed = set()

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        expansions = 0
        best_so_far = None
        best_f = float('inf')

        while open_set:
            f, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if expansions >= max_expansions:
                if best_so_far is not None:
                    return self._reconstruct_path(came_from, best_so_far)
                return None

            expansions += 1
            cg = g_score[current]
            for dr, dc in neighbors:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue

                raw_risk = self.grid[nr, nc]
                if not np.isfinite(raw_risk) or raw_risk >= risk_threshold:
                    continue

                move_cost = np.hypot(dr, dc)
                risk_norm = raw_risk / (self.max_risk + 1e-9)
                risk_term = 1.0 + self.risk_weight * (risk_norm ** self.risk_power)
                prox_term = 1.0 + self.prox_weight * self.proximity_cost[nr, nc]
                step_cost = move_cost * risk_term * prox_term

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
        self.f   = ca.Function('f', [states, controls], [rhs])

        # decision variables
        X = ca.SX.sym('X', 3, self.N+1)
        U = ca.SX.sym('U', 2, self.N)

        # parameters: [ x0(3), ref_x/ref_y (2*N), dt_seq (N) ]
        P = ca.SX.sym('P', 3 + 2*self.N + self.N + self.N + self.N)
        # idx helpers
        def idx_xr(k):   return 3 + 2*k
        def idx_yr(k):   return 3 + 2*k + 1
        def idx_dt(k):   return 3 + 2*self.N + k
        def idx_psir(k): return 3 + 2*self.N + self.N + k
        def idx_vr(k):   return 3 + 2*self.N + self.N + self.N + k

        # … constraints g, objective obj
        g = []; obj = 0
        g.append(X[:,0] - P[0:3])

        for k in range(self.N):
            st  = X[:,k]
            uc  = U[:,k]
            xr, yr  = P[idx_xr(k)],   P[idx_yr(k)]
            dt_k    = P[idx_dt(k)]
            psi_r   = P[idx_psir(k)]
            v_r     = P[idx_vr(k)]

            dx = xr - st[0]
            dy = yr - st[1]

            # Frenet-style errors
            e_ct  = -ca.sin(st[2])*dx + ca.cos(st[2])*dy
            e_at  =  ca.cos(st[2])*dx + ca.sin(st[2])*dy
            e_psi = ca.atan2(ca.sin(st[2]-psi_r), ca.cos(st[2]-psi_r))

            # weights (stronger cross-track & heading)
            w_ct, w_at, w_psi = 10.0, 0.2, 8.0
            obj += w_ct*e_ct**2 + w_at*e_at**2 + w_psi*e_psi**2

            # velocity tracking + effort
            obj += 0.8*(uc[0] - v_r)**2 + ca.mtimes([uc.T, self.R_u, uc]) * dt_k

            # rate penalties
            if k > 0:
                du = U[:,k] - U[:,k-1]
                obj += 0.2*du[0]**2 + 3.0*du[1]**2

            # dynamics
            fval = self.f(st, uc)  # store self.f earlier
            g.append(X[:,k+1] - (st + dt_k * fval))

        # terminal cost vs last stage refs
        xr_T = P[idx_xr(self.N-1)]
        yr_T = P[idx_yr(self.N-1)]
        psi_T= P[idx_psir(self.N-1)]
        err_pos = X[0:2,self.N] - ca.vertcat(xr_T, yr_T)
        err_psi = ca.atan2(ca.sin(X[2,self.N]-psi_T), ca.cos(X[2,self.N]-psi_T))
        Qf_pos = np.diag([25,25])
        obj += ca.mtimes([err_pos.T, Qf_pos, err_pos]) + 12.0*err_psi**2

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
        self._x_init = None
        self._u_init = None
        opts.update({
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 5e-3,
            'ipopt.linear_solver': 'mumps',
            'ipopt.warm_start_init_point': 'yes',
        })
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # bounds (X free, U in [0,V_max]×[-δ_max,δ_max])
        nX = 3*(self.N+1)
        self.lbx = [-ca.inf]*nX + [0.02, -self.delta_max]*self.N
        self.ubx = [ ca.inf]*nX + [self.V_max, self.delta_max]*self.N
        self.lbg = [0]*G.size1()
        self.ubg = [0]*G.size1()

    def solve(self, x0, ref_xy, ref_psi, ref_v, dt_seq):
        N = self.N
        assert len(ref_xy)==len(ref_psi)==len(ref_v)==len(dt_seq)==N

        if self._x_init is None:
            x_init = np.tile(x0, (N+1,1))
            u_init = np.zeros((N,2))
        else:
            x_init = self._x_init.copy()
            u_init = self._u_init.copy()

        P = np.concatenate([
            x0,
            ref_xy.reshape(-1),    # 2N
            np.asarray(dt_seq),    # N
            ref_psi,               # N
            ref_v                  # N
        ], axis=0)

        init = np.concatenate([x_init.ravel(), u_init.ravel()])
        sol  = self.solver(x0=init, lbx=self.lbx, ubx=self.ubx,
                        lbg=self.lbg, ubg=self.ubg, p=P)
        flat = sol['x'].full().ravel()
        Xopt = flat[:3*(N+1)].reshape(N+1,3)
        Uopt = flat[3*(N+1):].reshape(N,2)
        self._x_init, self._u_init = Xopt, Uopt
        return Uopt[0]  # (v, delta)

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

def in_edges(pt, x_edges, y_edges):
    return (x_edges[0] <= pt[0] <= x_edges[-1]) and (y_edges[0] <= pt[1] <= y_edges[-1])

def needs_recentering(vehicle_xy, dest_xy, x_edges, y_edges, buffer=1.0):
    x, y = vehicle_xy
    near_left   = x < x_edges[0] + buffer
    near_right  = x > x_edges[-1] - buffer
    near_bottom = y < y_edges[0] + buffer
    near_top    = y > y_edges[-1] - buffer
    dest_out    = not in_edges(dest_xy, x_edges, y_edges)
    return near_left or near_right or near_bottom or near_top or dest_out