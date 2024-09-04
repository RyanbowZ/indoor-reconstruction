import trimesh
import numpy as np
import os

name_list = ["apple", "banana", "cup","chair"]
name_idx = 3
name = name_list[name_idx]
suffix_list = ["obj","ply"]
suffix_idx = 0
suffix = suffix_list[suffix_idx]

# model_path = "/ghome/l6/yqliang/littleduck/datasets/fruits/models/apple/apple2.obj"
scene_name = "annex"
model_path = os.path.join(F"/ghome/l6/yqliang/littleduck/datasets/{scene_name}/models/", name + "_model."+suffix)





if __name__ == "__main__":
    model = trimesh.load(model_path, maintain_order=True, process = False)
    print(model.bounds)
    x_min = model.bounds[0][0]
    x_max = model.bounds[1][0]
    y_min = model.bounds[0][1]
    y_max = model.bounds[1][1]
    z_min = model.bounds[0][2]
    z_max = model.bounds[1][2]
    point_1 = [x_max, y_max, z_min]
    point_2 = [x_min,y_max, z_min]
    point_3 = [x_min, y_min, z_min]
    point_4 = [x_max, y_min, z_min]
    point_5 = [x_max,y_max,z_max]
    point_6 = [x_min, y_max, z_max]
    point_7 = [x_min, y_min, z_max]
    point_8 = [x_max, y_min, z_max]
    points = [point_1, point_2, point_3, point_4, point_5, point_6, point_7, point_8]
    points_path = os.path.join(os.path.dirname(model_path), f"{name}_points.npy")
    np.save(points_path, points)
    print("have saved points to: ", points_path)
    print(points)