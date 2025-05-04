import re, json, os, argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
def count_tracks(main_folder):
    try:
        new_path = rf'..\{main_folder}'
        pattern = re.compile(r'^track_\d+$')
        names = os.listdir(new_path)
        matches = [n for n in names if pattern.match(n)]
        return len(matches)
    except FileNotFoundError:
        print(f"Error: The directory '{new_path}' does not exist.")
        return 0
def quat2rot(q):
    qw, qx, qy, qz = q['w_val'], q['x_val'], q['y_val'], q['z_val']
    return np.array([
        [1-2*qy*qy-2*qz*qz,   2*qx*qy-2*qz*qw,     2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,     1-2*qx*qx-2*qz*qz,   2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,     2*qy*qz+2*qx*qw,     1-2*qx*qx-2*qy*qy]
    ])

def get_pose(state):
    p = state['kinematics_estimated']['position']
    o = state['kinematics_estimated']['orientation']
    pos = np.array([p['x_val'], p['y_val'], p['z_val']], dtype=float)
    return pos, quat2rot(o)


def label_points(main_folder, counter, start_num=0, threshold_check=1.2):
    # Go to sub folder and count tracks
    for i in tqdm(range(1, counter + 1), desc="Labeling tracks"):
        track_path = rf'..\{main_folder}\track_{i}'

        # Get the len of the files in the folder
        try:
            names = os.listdir(track_path)

            pattern_json = re.compile(r'^[^\.]+\.json$')
            names_json = [n for n in names if pattern_json.match(n)]
            
            pattern_png = re.compile(r'^[^\.]+\.png$')
            names_png = [n for n in names if pattern_png.match(n)]
            
            pattern_ply = re.compile(r'^[^\.]+\.ply$')
            names_ply = [n for n in names if pattern_ply.match(n)]

            # print(f'Found {len(names_json)} json in track {i}.')
            # print(f'Found {len(names_png)} png in track {i}.')
            # print(f'Found {len(names_ply)} ply in track {i}.')
            assert (len(names_json)//2) == len(names_png) == len(names_ply), f'Error: The number of json, png, and ply files do not match in track {i}.'

        except FileNotFoundError:
            print(f"Error: The directory '{track_path}' does not exist.")


        start_frame = start_num
        end_frame   = len(names_png)
        threshold   = threshold_check

        # label every points in every frame with previous and future frames
        for idx in tqdm(range(start_frame, end_frame), desc="Labeling frames"):
            plyf = track_path + f"/{idx}.ply"
            carf = track_path + f"/{idx}_car_state.json"
            collf= track_path + f"/{idx}_collision_info.json"
            if not (os.path.exists(plyf) and os.path.exists(carf) and os.path.exists(collf)):
                print(f"[skip {idx}] missing base files")
                continue

            cloud = o3d.io.read_point_cloud(plyf)
            pts = np.asarray(cloud.points)
            car0 = json.load(open(carf))
            pos0, R0 = get_pose(car0)
            world_pts = (R0 @ pts.T).T + pos0

            ever_close = np.zeros(len(world_pts), dtype=bool)
            ever_collision = np.zeros(len(world_pts), dtype=bool)

            j = 1
            while True:
                car_jf = track_path + f"/{j}_car_state.json"
                coll_jf = track_path + f"/{j}_collision_info.json"

                if not (os.path.exists(car_jf) and os.path.exists(coll_jf)):
                    break

                # load future car state & collision
                car_j = json.load(open(car_jf))
                coll_j = json.load(open(coll_jf))

                # update car‐proximity mask
                pos_car, _ = get_pose(car_j)
                d_car = np.linalg.norm(world_pts - pos_car, axis=1)
                ever_close |= (d_car < (threshold - 0.2))

                # update collision mask
                if coll_j['has_collided']:
                    cp = coll_j['position']
                    pos_col = np.array([cp['x_val'], cp['y_val'], cp['z_val']], dtype=float)
                    d_col = np.linalg.norm(world_pts - pos_col, axis=1)
                    ever_collision |= (d_col < (threshold + 0.2))
                j += 1

            #  0 = uncertainty, 1 = safe , 2 = risk
            labels = np.zeros(len(world_pts), dtype=np.uint8)
            labels[ ever_close & ~ever_collision ] = 1
            labels[ ever_collision ] = 2

            # make sure all points are labeled
            assert len(labels) == len(world_pts), \
                f"Label length mismatch: {len(labels)} vs {len(world_pts)}"

            out_file = os.path.join(track_path, f"{idx}_labels.npy")
            np.save(out_file, labels)

        # Check number of label files
        label_files = [f for f in os.listdir(track_path) if f.endswith('_labels.npy')]
        assert len(label_files) == len(names_ply), \
            f"Error: Number of label files does not match number of frames in track {i}."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='data', help='Main folder containing the tracks')
    parser.add_argument('--start_num', type=int, default=0, help='Start frame number')
    parser.add_argument('--threshold', type=float, default=1.2, help='Threshold for labeling')

    get_main_folder = parser.parse_args().folder
    get_start_num = parser.parse_args().start_num
    get_threshold = parser.parse_args().threshold

    label_points(get_main_folder, count_tracks(get_main_folder), get_start_num, get_threshold)
    
    print(f"Labeling completed for {get_main_folder} with start frame {get_start_num} and threshold {get_threshold}.")