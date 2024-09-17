import open3d as o3d
import numpy as np


def align_room(pcd, ground_normal, wall_normal):
    # Ensure the normals are unit vectors
    ground_normal = ground_normal / np.linalg.norm(ground_normal)
    wall_normal = wall_normal / np.linalg.norm(wall_normal)

    # Calculate the third axis using cross product
    third_axis = np.cross(ground_normal, wall_normal)
    third_axis = third_axis / np.linalg.norm(third_axis)

    # Create rotation matrix
    rotation_matrix = np.column_stack((wall_normal, third_axis, ground_normal))

    # Ensure it's a valid rotation matrix (determinant should be 1)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 0] *= -1

    # Apply rotation to the point cloud
    pcd.rotate(rotation_matrix.T, center=(0, 0, 0))

    return pcd


# Load your point cloud
pcd = o3d.io.read_point_cloud("./evans/mycloud.ply")
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# Define the normal vectors (example values, replace with your actual normals)
ground_normal = np.array([0.4729603259610954,0.142068458837462, 0.8695545313954283])  # Slightly off from [0, 0, 1]
wall_normal = np.array([0.8379942404025764, 0.1599403228536768, -0.5217132796638114])  # Slightly off from [1, 0, 0]
evans_ground = np.array([ 0.3515557967199977, 0.10683605453830, 0.930050847665467])
evans_wall = np.array([-0.06514, 0.9932, -0.09634])
# Align the room
aligned_pcd = align_room(pcd, evans_ground, evans_wall)

# Visualize the result
o3d.visualization.draw_geometries([aligned_pcd, origin])
o3d.io.write_point_cloud("./evans_aligned.ply", aligned_pcd)