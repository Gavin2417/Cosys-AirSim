import os
import time
import heapq

import cosysairsim as airsim
import numpy as np
import numpy.ma as ma
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap

import casadi as ca

from linefit import ground_seg
from function2 import calculate_combined_risks, compute_cvar_cellwise

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
        self.Q_psi = 0.1
        self.R_v = 0.1
        self.R_delta = 0.1

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
        opts = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0}
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
            lbx += [-self.V_max, -self.delta_max]
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
        u_init = np.zeros((self.N, 2))
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





# -------------- Utility & Planning Functions ---------------- #
def interpolate_in_radius(grid, radius):
    valid = ~np.isnan(grid)
    coords = np.column_stack(np.where(valid))
    values = grid[valid]
    tree = cKDTree(coords)
    nan_coords = np.column_stack(np.where(np.isnan(grid)))
    for coord in nan_coords:
        nbrs = tree.query_ball_point(coord, radius)
        if not nbrs:
            continue
        w_vals = []
        wts    = []
        for i in nbrs:
            d = np.linalg.norm(coord - coords[i]) + 1e-6
            w = 1.0/d
            wts.append(w)
            w_vals.append(w * values[i])
        grid[coord[0], coord[1]] = sum(w_vals)/sum(wts)
    return grid


def filter_points_by_radius(points, center, radius):
    d = np.linalg.norm(points[:, :2] - center, axis=1)
    return points[d <= radius]

# A* helpers
def is_valid(r,c,g):  return 0<=r<g.shape[0] and 0<=c<g.shape[1]
def is_unblocked(g,r,c): return not np.isnan(g[r,c]) and g[r,c]<1.0
def h(r,c,d):      return np.hypot(r-d[0], c-d[1])

def trace_path(parents, dest):
    path = []
    r,c = dest
    while True:
        path.append((r,c))
        pr,pc = parents[r,c]
        if (r,c)==(pr,pc): break
        r,c = pr,pc
    return path[::-1]

def a_star_search(risk, start, dest):
    R,C = risk.shape
    open_list = [(0.0, start)]
    g = np.full((R,C), np.inf)
    g[start] = 0
    f = np.full((R,C), np.inf)
    f[start] = h(*start, dest)
    parents = np.zeros((R,C,2), dtype=int)
    for i in range(R):
        for j in range(C): parents[i,j] = (i,j)

    while open_list:
        _, (r,c) = heapq.heappop(open_list)
        if (r,c)==dest:
            return trace_path(parents, dest)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if is_valid(nr,nc,risk) and is_unblocked(risk,nr,nc):
                tg = g[r,c] + risk[nr,nc]
                if tg < g[nr,nc]:
                    g[nr,nc] = tg
                    f[nr,nc] = tg + h(nr,nc,dest)
                    parents[nr,nc] = (r,c)
                    heapq.heappush(open_list, (f[nr,nc], (nr,nc)))
    return None


def smooth_path(path, window_size=5):
    p = np.array(path)
    n = len(p)
    if n<window_size: return p
    if window_size%2==0: window_size+=1
    hw = window_size//2
    sm = []
    for i in range(n):
        start = max(0,i-hw)
        end   = min(n, i+hw+1)
        sm.append(p[start:end].mean(axis=0))
    return np.array(sm)

# ----------------- Lidar & Grid Classes --------------------- #
class lidarTest:
    def __init__(self, lidar_name, vehicle_name):
        self.client = airsim.CarClient(ip="100.123.124.47")
        self.client.confirmConnection()
        self.vehicleName = vehicle_name
        self.lidarName = lidar_name
        self.lastlidarTimeStamp = 0

    def get_data(self, gpulidar=True):
        if gpulidar:
            d = self.client.getGPULidarData(self.lidarName, self.vehicleName)
        else:
            d = self.client.getLidarData(self.lidarName, self.vehicleName)
        if d.time_stamp==self.lastlidarTimeStamp:
            return None, None
        self.lastlidarTimeStamp = d.time_stamp
        if len(d.point_cloud)<2:
            return None, None
        pts = np.array(d.point_cloud, dtype=np.float32)
        dims = 5 if gpulidar else 3
        pts = pts.reshape(-1, dims)
        if not gpulidar:
            pts *= np.array([1, -1, 1])
        return pts, d.time_stamp

    def get_vehicle_pose(self):
        p = self.client.simGetVehiclePose()
        pos = np.array([p.position.x_val,
                        p.position.y_val,
                        p.position.z_val])
        q  = p.orientation
        R  = self.quaternion_to_rotation_matrix(q)
        return pos, R

    def quaternion_to_rotation_matrix(self, q):
        qw,qx,qy,qz = q.w_val, q.x_val, q.y_val, q.z_val
        return np.array([
            [1-2*qy*qy-2*qz*qz,   2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz,     2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy],
        ])
    def transform_to_world(self, points, position, rotation_matrix):
        points_rotated = np.dot(points, rotation_matrix.T)
        return points_rotated + position

class GridMap:
    def __init__(self, resolution):
        self.resolution = resolution
        self.grid = {}
    def get_cell(self, x,y):
        return round(int(x/self.resolution),1), round(int(y/self.resolution),1)
    def add_point(self, x,y,z,_):
        c = self.get_cell(x,y)
        self.grid.setdefault(c, []).append(z)
    def get_estimate(self):
        out = []
        for (gx,gy), zs in self.grid.items():
            out.append([gx*self.resolution, gy*self.resolution, np.mean(zs)])
        return np.array(out)

# ------------------------ Main Loop ------------------------- #
def main():
   # tune these if needed:
    nmpc = NMPCController(
        horizon=5,      # how many steps to look ahead
        dt=0.1,          # step time [s]
        wheelbase=0.25,   # Husky wheelbase [m]
        V_max=0.4        # max speed [m/s]
    )
    # AirSim & Lidar
    lidar_test = lidarTest('gpulidar1', 'CPHusky')
    lidar_test.client.enableApiControl(True, 'CPHusky')

    # Grid maps & segmentation
    ground_map   = GridMap(0.1)
    obstacle_map = GridMap(0.1)
    base = os.path.dirname(__file__)
    cfg  = os.path.join(base, '../assets/config.toml')
    if os.path.exists(cfg): groundseg = ground_seg(cfg)
    else:
        print(f"No config at {cfg}, using defaults.")
        groundseg = ground_seg()

    # Plot setup
    fig, ax = plt.subplots()
    plt.ion()
    cbar = None

    # Manual world path
    manual_world = [(-1,0), (0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
                    (6,0), (7,0), (8,0), (8,-1),(8,-2),(8,-3),(8,-4),(8,-5),(9,-5),(10,-5),(10,-6)]
    ctr = airsim.CarControls()
    try:
        while True:
            pts, ts = lidar_test.get_data(gpulidar=True)
            if pts is None: continue

            # World coords
            pos, R = lidar_test.get_vehicle_pose()
            pts_w = lidar_test.transform_to_world(pts[:,:3], pos, R)
            pts_w[:,2] *= -1
            labels = np.array(groundseg.run(pts_w))

            # Populate maps
            for p,lab in zip(pts_w, labels):
                if lab==1:      ground_map.add_point(*p, ts)
                elif p[2] > -pos[2]: obstacle_map.add_point(*p, ts)
                else:           ground_map.add_point(*p, ts)

            gpts = filter_points_by_radius(ground_map.get_estimate(), pos[:2], 15)
            x, y, z = gpts.T

            # Grid for risks
            start = np.array([-1,0]); dest = np.array([10,-5]); m=0.1; M=1
            mins = np.minimum(start,dest)-1; maxs = np.maximum(start,dest)+1
            xe = np.arange(mins[0], maxs[0]+m, m)
            ye = np.arange(mins[1], maxs[1]+m, m)
            xm = (xe[:-1]+xe[1:])/2; ym = (ye[:-1]+ye[1:])/2
            X,Y = np.meshgrid(xm, ym)
            Zg = np.full((len(xm),len(ym)), np.nan)
            for xi, yi, zi in zip(x,y,z):
                i = np.digitize(xi, xe)-1; j = np.digitize(yi, ye)-1
                if 0<=i<len(xm) and 0<=j<len(ym): Zg[i,j]=zi

            sr, sop = calculate_combined_risks(Zg, np.argwhere(~np.isnan(Zg)),
                                              max_height_diff=0.05,
                                              max_slope_degrees=30.0,
                                              radius=0.5)
            tot = np.ma.mean([ma.masked_invalid(sr), ma.masked_invalid(sop)], axis=0).filled(np.nan)

            # obstacles
            obst = filter_points_by_radius(obstacle_map.get_estimate(), pos[:2], 15)
            for ox, oy, _ in obst:
                i = np.clip(np.digitize(ox, xe)-1, 0, len(xm)-1)
                j = np.clip(np.digitize(oy, ye)-1, 0, len(ym)-1)
                tot[i,j] = 1.0

            # interpolate
            tot = interpolate_in_radius(tot, 1.5)
            cvar = compute_cvar_cellwise(ma.masked_invalid(tot), alpha=0.8)
            dist = np.hypot((X-pos[0]), (Y-pos[1]))
            cvar[dist.T>13] = np.nan

            # plot risk
            ax.clear()
            cmap = LinearSegmentedColormap.from_list('risk', [(0.5,0.5,0.5),(1,1,0),(1,0,0)])
            pcm = ax.pcolormesh(xm, ym, cvar.T, shading='auto', cmap=cmap, alpha=0.7)
            if cbar is None: cbar = fig.colorbar(pcm, ax=ax, label='Risk')
            else: cbar.update_normal(pcm)
            ax.set_aspect('equal')
            smooth = smooth_path(np.array(manual_world), window_size=3)

            ax.plot(smooth[:,0], smooth[:,1], '--m', lw=2, label='Path')
            ax.scatter(pos[0], pos[1], c='g', label='Start')
            ax.scatter(dest[0], dest[1], c='r', label='Goal')
            ax.legend()

            # Distance to goal
            dist_to_goal = np.hypot(dest[0] - pos[0], dest[1] - pos[1])
            if dist_to_goal <0.5:
                print("Goal reached!")
                ctr.throttle = 0.0
                ctr.steering = 0.0
                lidar_test.client.setCarControls(ctr)
                break
            mw = np.array(smooth)
            dists = np.linalg.norm(mw - pos[:2], axis=1)
            i_closest = int(np.argmin(dists))

            pts_list = smooth.tolist()    # now a Python list of [x,y]
            ref_list = pts_list[i_closest : i_closest + nmpc.N]

            if len(ref_list) < nmpc.N:
                ref_list += [pts_list[-1]] * (nmpc.N - len(ref_list))

            ref_traj = np.array(ref_list)

            # get current pose
            pos, R = lidar_test.get_vehicle_pose()
            psi = np.arctan2(R[1,0], R[0,0])  # extract yaw from rotation

            # current state
            x0 = [pos[0], pos[1], psi]
            if i_closest + 1 < len(smooth):
                next_wp = smooth[i_closest + 1]
            else:
                next_wp = smooth[-1]

            # compute desired heading to that waypoint
            dx, dy = next_wp - pos[:2]
            desired_psi = np.arctan2(dy, dx)
            # wrap error to [-pi,pi]
            angle_err = np.arctan2(np.sin(desired_psi - psi),
                                np.cos(desired_psi - psi))

            # if it’s a “big turn,” just steer in place
            big_turn_threshold = np.deg2rad(30)   # e.g. 20
            delta_max = np.deg2rad(30)
            if abs(angle_err) > big_turn_threshold:
                # stop forward motion
                ctr.throttle = -0.2
                # steering control: map angle_err to [-1,1]
                ctr.steering = float(np.clip(angle_err / delta_max, -1.0, 1.0))
                lidar_test.client.setCarControls(ctr)
                plt.pause(0.1)
                continue  # skip the NMPC step entirely

            # otherwise, run NMPC as before
            v_cmd, delta_cmd = nmpc.solve(x0, ref_traj)
            ctr.throttle = float(np.clip(v_cmd / nmpc.V_max,   -1.0, 1.0))
            ctr.steering = float(np.clip(delta_cmd / nmpc.delta_max, -1.0, 1.0))
            lidar_test.client.setCarControls(ctr)

            plt.draw(); plt.pause(0.1)

    finally:
        plt.ioff(); plt.show()

if __name__ == '__main__':
    main()
