import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay


def save_point_cloud_as_obj(point_cloud, filename):
    # Convert point cloud to mesh (vertices only, no faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = point_cloud.points

    # If the point cloud has colors, convert them to 8-bit RGB
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        colors_uint8 = (colors * 255).astype(np.uint8)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_uint8)

    # Save as OBJ
    o3d.io.write_triangle_mesh(filename, mesh)

def remove_points_in_hexahedron(point_cloud, hexahedron_points):
    # Convert hexahedron points to numpy array
    hexahedron_points = np.asarray(hexahedron_points)

    # Create a ConvexHull from the hexahedron points
    hull = Delaunay(hexahedron_points)

    # Get points from the point cloud
    points = np.asarray(point_cloud.points)

    # Check which points are inside the convex hull
    in_hull = hull.find_simplex(points) >= 0

    # Create a new point cloud with points outside the hexahedron
    filtered_points = points[~in_hull]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        filtered_colors = colors[~in_hull]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # If the original point cloud has normals, preserve them
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        filtered_normals = normals[~in_hull]
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_pcd

pcd = o3d.io.read_point_cloud("rooms_full.ply")
apple_npy = np.load('apple_corners_mean.npy')
banana_npy = np.load('banana_corners_mean.npy')
cup_npy = np.load('cup_corners_mean.npy')
filtered_pcd = remove_points_in_hexahedron(pcd, apple_npy[:8])
filtered_pcd = remove_points_in_hexahedron(filtered_pcd, banana_npy[:8])
filtered_pcd = remove_points_in_hexahedron(filtered_pcd, cup_npy[:8])
o3d.io.write_point_cloud("remove_apple_banana_cup.ply", filtered_pcd)

apple = o3d.io.read_triangle_mesh("apple2.obj")
banana = o3d.io.read_triangle_mesh("banana_model.obj")
cup = o3d.io.read_triangle_mesh("cup_rotated.obj")

apple_matrix = [[ 0.10257509, 0.02006307,  0.02389645,  0.48510218],
 [-0.03469795,  0.11952283,  0.03572152,  0.23416159],
 [-0.02308838, -0.0904882,   0.05248169, -0.74064908],
 [ 0,         0,         0,        1       ]]

banana_matrix =  [[ 1.83768704e-03, -1.37959188e-02,  2.36471393e-03,  4.95076911e-01],
 [ 1.91336124e-02, -3.64082156e-04, -3.86905522e-03,  7.99067835e-02],
 [ 8.49400054e-03,  3.80489593e-03,  8.20383735e-03, -7.50430251e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

cup_matrix = [[ 0.02662029, -0.04249567,  0.0034949,   0.39083337],
 [ 0.01223236,  0.08551654,  0.00498392,  0.51473996],
 [-0.00687299, -0.0123932,   0.02240661, -0.71382303],
 [ 0,         0,         0,        1       ]]

apple.transform(apple_matrix)
banana.transform(banana_matrix)
# cup.transform(cup_matrix)

o3d.visualization.draw_geometries([filtered_pcd, apple, banana, cup])
