import open3d as o3d
import numpy as np
import os

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

    return pcd, rotation_matrix


def restore_original_orientation(pcd, rotation_matrix):
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    return pcd


# Load your point cloud
pcd = o3d.io.read_point_cloud("./716/reconsimple.ply")
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)



# Define the normal vectors (example values, replace with your actual normals)
scan716_ground = np.array([-0.05955520356774661, -0.8407159371950883, -0.53819131419430099])
scan716_wall = np.array([-0.17586655897933534, -0.5496843009148765, 0.8166505511909554])

# Align the room and get the rotation matrix
aligned_pcd, rotation_matrix = align_room(pcd, scan716_ground, scan716_wall)

print("rotation_matrix", rotation_matrix)

# Restore the original orientation
restored_pcd = restore_original_orientation(aligned_pcd, rotation_matrix)

pcl_dir = '716/group/denoise/'
for file in os.listdir(pcl_dir):
    if file.endswith('.ply'):
        pcl_path = os.path.join(pcl_dir, file)
        point_cloud = o3d.io.read_point_cloud(pcl_path)
        restore_pcl_pcd = restore_original_orientation(point_cloud, rotation_matrix)
        o3d.io.write_point_cloud(f'716/group/rotate_back/{file}', restore_pcl_pcd)


# Visualize the restored point cloud
# o3d.visualization.draw_geometries([restored_pcd, origin])
