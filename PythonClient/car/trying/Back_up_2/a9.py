# import setup_path
import os, math, time, heapq
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, norm
from scipy.ndimage import distance_transform_edt
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

class NMPCController:
    def __init__(self, horizon, wheelbase, V_max, delta_max,
                 Q_x=1.0, Q_y=1.0, Q_psi=1.0, Q_risk=5.0,
                 R_v=0.1, R_delta=0.1):
        self.N = horizon
        self.L = wheelbase
        self.V_max = V_max
        self.delta_max = delta_max
        self.Q_x = Q_x
        self.Q_y = Q_y
        self.Q_psi = Q_psi
        self.Q_risk = Q_risk
        self.R_v = R_v
        self.R_delta = R_delta
        self.Q_goal  = 10.0
        
        x = ca.SX.sym('x'); y = ca.SX.sym('y'); psi = ca.SX.sym('psi')
        states = ca.vertcat(x, y, psi); n_states = states.size1()
        v = ca.SX.sym('v'); delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta); n_controls = controls.size1()

        rhs = ca.vertcat(v*ca.cos(psi), v*ca.sin(psi), v/self.L * ca.tan(delta))
        f = ca.Function('f', [states, controls], [rhs])

        X = ca.SX.sym('X', n_states, self.N+1)
        U = ca.SX.sym('U', n_controls, self.N)
        OPT = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        P = ca.SX.sym('P', n_states + 2*self.N + self.N + self.N)
        idx = 0
        idx_x0   = idx; idx += n_states
        idx_ref  = idx; idx += 2*self.N
        idx_dt   = idx; idx += self.N
        idx_risk = idx

        obj = 0
        g   = []
        lbg = []
        ubg = []

        # 1) initial‐state equality:  X[:,0] == x0
        g.append( X[:,0] - P[idx_x0:idx_x0+n_states] )
        lbg += [0]*n_states
        ubg += [0]*n_states
        goal_x = P[idx_ref + 2*(self.N-1)]
        goal_y = P[idx_ref + 2*(self.N-1) + 1]
        for k in range(self.N):
            rx = P[idx_ref + 2*k]
            ry = P[idx_ref + 2*k + 1]
            st = X[:,k]
            con = U[:,k]

            ang_des = ca.atan2(ry - st[1], rx - st[0])
            ang_err = ca.atan2(ca.sin(st[2] - ang_des), ca.cos(st[2] - ang_des))
            obj += self.Q_x*(st[0] - rx)**2 + self.Q_y*(st[1] - ry)**2 + self.Q_psi*ang_err**2

            obj += self.R_v*(con[0]/self.V_max)**2 + self.R_delta*(con[1]/self.delta_max)**2
            obj += self.Q_goal * ((st[0] - goal_x)**2 + (st[1] - goal_y)**2)
            if k > 0:
                du = U[:,k] - U[:,k-1]
                obj += 0.05*ca.sumsqr(du)

            risk_k = P[idx_risk + k]
            obj += self.Q_risk * risk_k

            dt_k = P[idx_dt + k]
            st_next = X[:,k+1]
            fval = f(st, con)
            g.append( st_next - (st + dt_k * fval) )
            lbg += [0]*n_states
            ubg += [0]*n_states
            g.append( risk_k )
            lbg.append( -ca.inf )             # no lower bound
            ubg.append(0.4)     
        rxT = P[idx_ref + 2*(self.N-1)]
        ryT = P[idx_ref + 2*(self.N-1) + 1]
        psi_ref = ca.atan2(ryT - X[1,self.N-1], rxT - X[0,self.N-1])
        Qf = ca.diag(ca.SX([20, 20, 5]))
        errT = X[:,self.N] - ca.vertcat(rxT, ryT, psi_ref)
        obj += errT.T @ Qf @ errT

        g = ca.vertcat(*g)
        nlp = {'f': obj, 'x': OPT, 'g': g, 'p': P}
        opts = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': False}
        self.solver = ca.nlpsol('solver','ipopt',nlp,opts)

        self.lbx = [-ca.inf]*(n_states*(self.N+1)) + [0, -self.delta_max]*self.N
        self.ubx = [ ca.inf]*(n_states*(self.N+1)) + [self.V_max, self.delta_max]*self.N
        self.lbg = lbg
        self.ubg = ubg

    def solve(self, x0, ref_traj, dt_seq, risk_seq):
        p = np.concatenate([x0,
                            ref_traj.reshape(-1),
                            np.array(dt_seq),
                            np.array(risk_seq)])
        x_init = np.tile(x0, (self.N+1, 1))
        u_init = np.zeros((self.N, 2))
        init = np.concatenate([x_init.flatten(), u_init.flatten()])

        sol = self.solver(x0=init,
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
        return np.array(estimates)\

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

    def plan(self, start, goal):
        # check validity
        for pt in (start, goal):
            r,c = pt
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if self.cost_map[r,c] == np.inf:
                return None

        open_set = []
        g_score = {start: 0.0}
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        came_from = {}

        # 8‑connected
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]

        while open_set:
            f, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            cg = g_score[current]
            for dr, dc in neighbors:
                nr, nc = current[0]+dr, current[1]+dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                step_cost = self.cost_map[nr, nc] * np.hypot(dr, dc)
                if step_cost == np.inf:
                    continue
                tentative = cg + step_cost
                neighbor = (nr, nc)
                if tentative < g_score.get(neighbor, np.inf):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative + self._heuristic(neighbor, goal),
                                              neighbor))

        return None
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
    half = window_size // 2
    sm = [np.mean(path[max(0, i-half):min(n_points, i+half+1)], axis=0)
          for i in range(n_points)]
    return np.array(sm)

if __name__ == '__main__':
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    # lidar_test.client.enableApiControl(True, 'CPHusky')
    seg = RandlaGroundSegmentor(device=None, subsample_grid=0.1)

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
    N = 20
    # nmpc = NMPCController(horizon=N,
    #                       wheelbase=0.25,
    #                       V_max=0.05,
    #                       delta_max=np.deg2rad(25),
    #                       Q_risk=500.0)
    ctr = airsim.CarControls()

    cmap = LinearSegmentedColormap.from_list("gray_yellow_red",
               [(0.5,0.5,0.5),(1,1,0),(1,0,0)], N=10)
    fig, ax = plt.subplots(); plt.ion(); prev_t = time.time()
    grid_map = {}
    grid_map_ground = GridMap(resolution=0.1)
    colorbar = None
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
            veh_xy = np.array([vehicle_y, vehicle_x])  # (y, x) for plotting
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
            valid_indices = np.argwhere(~np.isnan(risk_grid))
            if valid_indices.size > 0:
                candidate_centers = np.column_stack((x_mid[valid_indices[:, 0]], y_mid[valid_indices[:, 1]]))
                candidate_distances = np.linalg.norm(candidate_centers - destination_point, axis=1)
                best_candidate = valid_indices[np.argmin(candidate_distances)]
                dest_idx = tuple(best_candidate)
                temp_dest = (x_edges[dest_idx[0]], y_edges[dest_idx[1]])
            # 3) Plan with A* Planner
            planner = AStarPlanner(risk_grid)
            start_idx = (np.digitize(vehicle_x, x_edges) - 1,
                         np.digitize(vehicle_y, y_edges) - 1)
            goal_idx = (np.digitize(temp_dest[0], x_edges) - 1,
                        np.digitize(temp_dest[1], y_edges) - 1)
            path_idx = planner.plan(start_idx, goal_idx)

            # Convert path to world coords
            

            # Visualization
            ax.clear()
            c = ax.pcolormesh(Y, X, risk_grid.T, shading='auto', cmap=cmap, alpha=0.7)
            if colorbar is None:
                colorbar = fig.colorbar(c, ax=ax, label='Risk')
            else:
                colorbar.update_normal(c)
            ax.scatter(vehicle_y, vehicle_x, c='green', s=50, label='Vehicle')
            ax.scatter(destination_point[1], destination_point[0], c='red', s=50, label='Goal')
            ax.scatter(temp_dest[1], temp_dest[0], c='red', s=50, label='Goal')
            if path_idx:
                raw_coords = np.array([[x_mid[r], y_mid[c]] for r, c in path_idx])
                smoothed = smooth_path(raw_coords, window_size=5)
                ax.plot(smoothed[:,1],
                        smoothed[:,0],
                        color='blue',
                        linewidth=2,
                        label='Smoothed A* Path')
            ax.legend()
            plt.draw(); plt.pause(0.1)

            # 4) Control or exit
            if np.hypot(vehicle_x - destination_point[0], vehicle_y - destination_point[1]) < 0.75:
                lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0))
                break



            # now = time.time(); dt = max(min(now-prev_t, 0.2), 0.1); prev_t = now
            # dt_seq = [dt]*N
            # psi0 = math.atan2(R[1,0], R[0,0])      # extract yaw from rotation matrix
            # u_cmd = nmpc.solve([pos[0], pos[1], psi0], ref_traj, dt_seq, risk_seq)

            # ctr.throttle = float(np.clip(u_cmd[0]/nmpc.V_max, 0, 0.02))
            # ctr.steering = float(np.clip(u_cmd[1]/nmpc.delta_max, -1, 1))
            # lidar_test.client.setCarControls(ctr)

            # ax.clear()
            # c = ax.pcolormesh(Y, X, risk_grid_t.T, shading='auto', cmap=cmap, alpha=0.7)
            # ax.scatter(veh_xy[0], veh_xy[1], c='green', s=50, label='Vehicle')
            # ax.scatter(destination_point[1], destination_point[0], c='red', s=50, label='Goal')
            # if colorbar is None:
            #     colorbar = fig.colorbar(c, ax=ax, label='Risk')
            # else:
            #     colorbar.update_normal(c)
            # line = np.vstack((veh_xy, temp_dest))
            # #  plot ref _traj
            # ax.plot(ref_traj[:,1], ref_traj[:,0], 'r-', label='Ref Traj')
            # # ax.plot(line[:,1], line[:,0], 'b--', label='Ref')
            # ax.legend(); plt.pause(0.05)

            # if np.linalg.norm(veh_xy - destination_point) < 0.75:
            #     lidar_test.client.setCarControls(airsim.CarControls(throttle=0, steering=0))
            #     break
    finally:
        plt.ioff(); plt.show()
