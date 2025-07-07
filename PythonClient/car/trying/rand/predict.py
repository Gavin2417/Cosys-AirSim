import torch
import numpy as np
from sklearn.neighbors import KDTree
from network.RandLANet import Network
from utils.config import ConfigSemanticKITTI as cfg
from utils.data_process import DataProcessing as DP

class RandlaGroundSegmentor:
    def __init__(self, ckpt_path='log/checkpoint_old.tar', device=None, subsample_grid=0.1):
        """
        ckpt_path: path to your trained RandLA-Net checkpoint .pth
        device: torch.device or None to auto-select CUDA if available
        subsample_grid: voxel size for occasional downsampling
        """
        # device setup
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model
        self.model = Network(cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        # settings
        self.num_points = cfg.num_points
        self.subsample_grid = subsample_grid
        self.k_n = cfg.k_n
        self.num_layers = cfg.num_layers
        self.sub_ratio = cfg.sub_sampling_ratio

    def segment(self, points_world):
        """
        points_world: (M,3) np.array of live LiDAR points in world coords
        returns: labels_world (M,) ints 1=ground, 0=non-ground
        """
        
        M = points_world.shape[0]

        # 1) conditional voxel‐grid subsample if too dense
        pts = DP.preprocess_point_cloud(points_world,
                                    grid_size=self.subsample_grid,
                                    nb_neighbors=20,
                                    std_ratio=2.0,
                                    target_n=self.num_points)
        
        # 2) enforce exactly num_points via random trunc/pad
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            dup = np.random.choice(N, self.num_points - N, replace=True)
            idx = np.concatenate([np.arange(N), dup])
        sampled = pts[idx]

        # 3) build per-layer xyz lists and features
        xyz_list = [torch.from_numpy(sampled)
                        .float()
                        .to(self.device)
                        .unsqueeze(0)]          # [1, N, 3]
        feat = xyz_list[0].permute(0, 2, 1).contiguous()  # [1, 3, N]

        neigh_idx, sub_idx, interp_idx = [], [], []
        pts_cur = sampled
        for i in range(self.num_layers):
            # knn at this scale
            knn = DP.knn_search(pts_cur[None, ...], pts_cur[None, ...], self.k_n)[0]
            neigh_idx.append(torch.from_numpy(knn)
                                  .long()
                                  .to(self.device)
                                  .unsqueeze(0))
            # subsample
            n_sub = pts_cur.shape[0] // self.sub_ratio[i]
            sel = np.random.permutation(pts_cur.shape[0])[:n_sub]
            sub_idx.append(torch.from_numpy(sel[None, ...])
                                .long()
                                .to(self.device))
            # build next level pts
            pts_next = pts_cur[sel]
            xyz_list.append(torch.from_numpy(pts_next)
                                 .float()
                                 .to(self.device)
                                 .unsqueeze(0))
            # interp indices back
            interp = DP.knn_search(pts_next[None,...], pts_cur[None,...], 1)[0]
            interp_idx.append(torch.from_numpy(interp)
                                   .long()
                                   .to(self.device)
                                   .unsqueeze(0))

            pts_cur = pts_next

        # 4) inference
        with torch.no_grad():
            inputs = {
                "xyz":        xyz_list,
                "features":   feat,
                "neigh_idx":  neigh_idx,
                "sub_idx":    sub_idx,
                "interp_idx": interp_idx
            }
            end_pts = self.model(inputs)
            preds = end_pts["logits"].argmax(dim=1).squeeze(0).cpu().numpy()

        # 5) project back to original M points
        tree = KDTree(sampled)
        nn = tree.query(points_world, return_distance=False).squeeze(-1)
        labels_world = preds[nn]
        return labels_world
import open3d as o3d
point_path = "../data/1/1.ply"
pcd = o3d.io.read_point_cloud(point_path)
points_np = np.asarray(pcd.points)
print(len(points_np))
# Segment ground vs non-ground
seg = RandlaGroundSegmentor()
labels = seg.segment(points_np)

# Assign color: ground = green, non-ground = red
colors = np.zeros_like(points_np)
colors[labels == 1] = [0, 1, 0]   # green for ground
colors[labels == 0] = [1, 0, 0]   # red for non-ground
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])