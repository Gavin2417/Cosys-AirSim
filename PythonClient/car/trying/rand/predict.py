import torch
import numpy as np
from sklearn.neighbors import KDTree
from network.RandLANet import Network
from utils.config import ConfigSemanticKITTI as cfg
from utils.data_process import DataProcessing as DP
import open3d as o3d

class RandlaGroundSegmentor:
    def __init__(self, ckpt_path='log/checkpoint_original.tar', device=None, subsample_grid=0.1):
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

    def segment(self, points):
        """
        points: (N,3) numpy array of the full-resolution cloud (no labels required)
        returns:
        labels: (N,) int array of predicted class indices
        """
        # 1) Subsample the full cloud
        sub_points = DP.grid_sub_sampling(points, grid_size=self.subsample_grid)

        # 2) Optionally remove outliers (statistical outlier removal)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub_points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        sub_points = np.asarray(pcd.points)

        # 3) Enforce exactly num_points via random truncation or duplication
        N = sub_points.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            dup = np.random.choice(N, self.num_points - N, replace=True)
            idx = np.concatenate([np.arange(N), dup])
        sampled = sub_points[idx]

        # 4) Build per-layer xyz lists and features
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
                              .to(self.device)
                              .unsqueeze(-1))
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

        # 5) Inference
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

        # 6) Project back to original points
        tree = KDTree(sampled)
        nn = tree.query(points, return_distance=False).squeeze(-1)
        labels = preds[nn]
        return labels

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d

# # 1) instantiate model ONCE
# seg = RandlaGroundSegmentor()

# # 2) set up interactive plotting
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(1,150):
#     point_path = f"../data/4/{i}.ply"
#     pcd = o3d.io.read_point_cloud(point_path)
#     points_np = np.asarray(pcd.points, dtype= np.float32)
#     print(f"Frame {i}, points: {len(points_np):,}")

#     # 3) segment once per frame
#     labels = seg.segment(points_np)
#     print("Unique labels:", np.unique(labels))

#     # 4) color array
#     colors = np.zeros_like(points_np)
#     colors[labels == 1] = [0, 1, 0]
#     colors[labels == 0] = [1, 0, 0]

#     # 5) clear the old scatter and draw new one
#     ax.clear()
#     ax.scatter(
#         points_np[:,0], points_np[:,1], points_np[:,2],
#         c=colors,
#         s=0.5,
#         depthshade=False
#     )
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'Ground vs. Non-Ground (frame {i})')

#     # 6) redraw & pause briefly
#     plt.draw()
#     plt.pause(0.1)

# # finish
# plt.ioff()
# plt.show()
