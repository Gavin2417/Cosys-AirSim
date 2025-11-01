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


import numpy as np
from scipy.ndimage import convolve

def calculate_combined_risks_plane(
    Z_grid,
    grid_resolution,                 # cell size [m]
    window_radius_m=0.3,             # plane-fit half-window in meters
    max_slope_degrees=30.0,          # slope risk cap
    max_step_height=0.4,             # step risk cap (height jump)
    min_points=6                     # minimum valid samples in window
):
    """
    Plane-fit slope + neighbor step risks (same output shape as Z_grid).
    - Slope: fit z = ax*x + ay*y + c in a (2r+1)^2 window via normal equations.
             Risk = min( sqrt(ax^2+ay^2) / tan(max_slope), 1 ).
    - Step:  max absolute height jump to 8-neighbors, normalized by max_step_height.
    NaNs are handled; windows with < min_points → slope = NaN.
    """

    Z = np.asarray(Z_grid, dtype=float)
    H, W = Z.shape
    cell = float(grid_resolution)

    # ----- coordinate grids in meters (centered at (0,0) in world frame OK) -----
    # absolute coords aren't required; only relative within the window matter.
    # Using array indices scaled by cell size is enough.
    yy, xx = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    X = xx * cell
    Y = yy * cell

    # ----- window kernel -----
    r = max(1, int(round(window_radius_m / cell)))
    k = 2 * r + 1
    K = np.ones((k, k), dtype=float)

    # ----- valid mask & masked fields -----
    M = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z, nan=0.0)

    # sums over window (via convolution)
    n   = convolve(M, K, mode='constant', cval=0.0)
    Sx  = convolve(X * M, K, mode='constant', cval=0.0)
    Sy  = convolve(Y * M, K, mode='constant', cval=0.0)
    Sz  = convolve(Z0 * M, K, mode='constant', cval=0.0)

    Sxx = convolve((X * X) * M, K, mode='constant', cval=0.0)
    Syy = convolve((Y * Y) * M, K, mode='constant', cval=0.0)
    Sxy = convolve((X * Y) * M, K, mode='constant', cval=0.0)

    Sxz = convolve((X * Z0) * M, K, mode='constant', cval=0.0)
    Syz = convolve((Y * Z0) * M, K, mode='constant', cval=0.0)

    # means in window
    with np.errstate(invalid='ignore', divide='ignore'):
        invn = np.where(n > 0, 1.0 / n, 0.0)
        # central (demeaned) moments (Σ x'^2 etc.)
        A = Sxx - (Sx * Sx) * invn            # Σ x'^2
        C = Syy - (Sy * Sy) * invn            # Σ y'^2
        B = Sxy - (Sx * Sy) * invn            # Σ x'y'
        Xz = Sxz - (Sx * Sz) * invn           # Σ x'z'
        Yz = Syz - (Sy * Sz) * invn           # Σ y'z'

    # solve 2x2 normal equations per cell:
    # [A B][ax] = [Xz]
    # [B C][ay]   [Yz]
    det = A * C - B * B
    eps = 1e-12
    good = (n >= min_points) & np.isfinite(det) & (np.abs(det) > eps)

    ax = np.full_like(Z, np.nan, dtype=float)
    ay = np.full_like(Z, np.nan, dtype=float)
    ax[good] = ( C[good] * Xz[good] - B[good] * Yz[good]) / det[good]
    ay[good] = (-B[good] * Xz[good] + A[good] * Yz[good]) / det[good]

    # slope gradient magnitude g = sqrt(ax^2 + ay^2)  (since tan(theta) = g)
    g = np.sqrt(ax * ax + ay * ay)

    # normalize by tan(max_slope)
    max_slope_rad = np.deg2rad(30)
    g_cap = np.tan(max_slope_rad)
    g_cap = max(g_cap, 1e-9)
    slope_risk = np.clip(g / g_cap, 0.0, 1.0)

    # restore NaNs where Z was NaN (optional; keeps map edges NaN)
    slope_risk[~np.isfinite(Z)] = np.nan
    def downhill_drop(Z, dx, dy, cell):
        H, W = Z.shape
        r0, r1 = max(0, dx), H + min(0, dx)
        c0, c1 = max(0, dy), W + min(0, dy)
        base  = Z[r0:r1, c0:c1]
        neigh = Z[r0-dx:r1-dx, c0-dy:c1-dy]
        # positive "drop" means base > neighbor (downhill toward neighbor)
        drop = base - neigh
        # convert to slope per meter along that direction
        horiz = np.hypot(dx, dy) * cell
        with np.errstate(invalid='ignore', divide='ignore'):
            slope = np.where(np.isfinite(drop), drop / max(horiz, 1e-9), 0.0)
        slope[~np.isfinite(base) | ~np.isfinite(neigh)] = 0.0
        # only keep *downhill* (positive drop)
        slope[slope < 0] = 0.0
        out = np.zeros_like(Z, dtype=float)
        out[r0:r1, c0:c1] = slope
        return out

    shifts = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    downhill_slopes = [downhill_drop(Z, dx, dy, cell) for dx, dy in shifts]
    max_downhill_slope = np.max(np.stack(downhill_slopes, axis=0), axis=0)

    # Normalize by the same cap as slope, or by a separate "max_step_height"
    # Here we normalize to a risk in [0,1] using tan(max_slope) for consistency:
    downhill_risk = np.clip(max_downhill_slope / g_cap, 0.0, 1.0)
    downhill_risk[~np.isfinite(Z)] = np.nan
    # ---------- step risk via 8-neighborhood height jumps (no wrap) ----------
    # neighbor diffs
    def neighbor_diff_no_wrap(Z, dx, dy):
        r0, r1 = max(0, dx), H + min(0, dx)
        c0, c1 = max(0, dy), W + min(0, dy)
        base  = Z[r0:r1, c0:c1]
        neigh = Z[r0-dx:r1-dx, c0-dy:c1-dy]
        diff = np.abs(base - neigh)
        invalid = ~np.isfinite(base) | ~np.isfinite(neigh)
        diff[invalid] = 0.0
        out = np.zeros_like(Z, dtype=float)
        out[r0:r1, c0:c1] = diff
        return out

    shifts = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    diffs = [neighbor_diff_no_wrap(Z, dx, dy) for dx, dy in shifts]
    max_diff = np.max(np.stack(diffs, axis=0), axis=0)

    step_risk = np.clip(max_diff / max_step_height, 0.0, 1.0)
    step_risk[~np.isfinite(Z)] = np.nan

    return step_risk, slope_risk, downhill_risk, ax, ay




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
        self.client = airsim.CarClient(ip="192.168.68.107")
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
            # inside for k in range(self.N):
            xr = P[3 + 2*k]
            yr = P[3 + 2*k + 1]
            dt_k = P[3 + 2*self.N + k]

            st = X[:,k]
            uc = U[:,k]

            # --- Frenet-style error (cross-track + heading) ---
            dx = xr - st[0]
            dy = yr - st[1]
            des_psi = ca.atan2(dy, dx)                     # path tangent
            # cross-track error (signed, in body frame)
            e_ct = -ca.sin(st[2])*dx + ca.cos(st[2])*dy
            # along-track error (optional, keep small weight)
            e_at =  ca.cos(st[2])*dx + ca.sin(st[2])*dy
            e_psi = ca.atan2(ca.sin(st[2] - des_psi), ca.cos(st[2] - des_psi))

            # weights (tune aggressively if you’re lagging)
            w_ct, w_at, w_psi = 8.0, 0.5, 6.0
            obj += w_ct*e_ct**2 + w_at*e_at**2 + w_psi*e_psi**2

            # control effort
            obj += ca.mtimes([uc.T, self.R_u, uc]) * dt_k

            # smoothness
            if k > 0:
                du = U[:,k] - U[:,k-1]
                obj += self.R_du * ca.sumsqr(du)

            # dynamics
            st_next = X[:,k+1]
            fval = f(st, uc)
            g.append(st_next - (st + dt_k * fval))

        # terminal cost
        errT = X[:,self.N] - ca.vertcat(
            P[3+2*(self.N-1)],
            P[3+2*(self.N-1)+1],
            0
        )
        Qf = np.diag([15, 15, 8])   # was [10,10,5]
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
        self.lbx = [-ca.inf]*nX + [0, -self.delta_max]*self.N
        self.ubx = [ ca.inf]*nX + [self.V_max, self.delta_max]*self.N
        self.lbg = [0]*G.size1()
        self.ubg = [0]*G.size1()

    def solve(self, x0, ref_traj, dt_seq):
        N = self.N
        assert len(dt_seq)==N
        if self._x_init is None:
            x_init = np.tile(x0, (N+1,1))
            u_init = np.zeros((N,2))
        else:
            x_init = self._x_init
            u_init = self._u_init
        p = np.concatenate([x0, ref_traj[:N].reshape(-1), np.array(dt_seq)])
        x_init = np.tile(x0, (N+1,1))
        u_init = np.zeros((N,2))
        init   = np.concatenate([x_init.flatten(), u_init.flatten()])

        sol = self.solver(x0=init,
                          lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg,
                          p=p)
        flat = sol['x'].full().ravel()
        U_opt = flat[-2*N:].reshape(N,2)
        X_opt = flat[:3*(N+1)].reshape(N+1,3)
        self._x_init = X_opt
        self._u_init = U_opt
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