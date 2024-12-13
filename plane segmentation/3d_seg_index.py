import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("./evans_aligned.ply")

vertices = np.asarray(pcd.points)

index = np.load("./seg_object_on_slam/2/indicator.npy")

selected_vertices = vertices[index]

seg_ply = o3d.geometry.PointCloud()
seg_ply.points = o3d.utility.Vector3dVector(selected_vertices)

if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    seg_ply.colors = o3d.utility.Vector3dVector(colors[index])

if pcd.has_normals():
    normals = np.asarray(pcd.normals)
    seg_ply.normals = o3d.utility.Vector3dVector(normals[index])

o3d.visualization.draw_geometries([seg_ply])