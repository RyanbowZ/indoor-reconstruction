import open3d as o3d
import numpy as np
from scipy.spatial import distance_matrix
import os
import glob
import re


def load_point_clouds(file_list):
    """Load multiple point clouds from a list of file paths."""
    return [o3d.io.read_point_cloud(file) for file in file_list]


def compute_centroids(point_clouds):
    """Compute centroids for a list of point clouds."""
    return np.array([np.mean(np.asarray(pc.points), axis=0) for pc in point_clouds])


def group_overlapping_point_clouds(point_clouds, threshold):
    """Group point clouds based on centroid proximity."""
    centroids = compute_centroids(point_clouds)
    n = len(point_clouds)

    # Compute pairwise distances between centroids
    dist_matrix = distance_matrix(centroids, centroids)

    # Initialize groups
    groups = [{i} for i in range(n)]

    # Merge groups based on centroid proximity
    for i in range(n):
        for j in range(i + 1, n):
            print (i, j, dist_matrix[i, j])
            if dist_matrix[i, j] < threshold:
                # Find the groups containing i and j
                group_i = next(group for group in groups if i in group)
                group_j = next(group for group in groups if j in group)

                # Merge the groups if they're different
                if group_i != group_j:
                    group_i.update(group_j)
                    groups.remove(group_j)

    return groups

rootpath = '716'
# Example usage
file_list = ["1_denoised.ply", "2_denoised.ply", "7_denoised.ply", "8_denoised.ply", "10_denoised.ply", "11_denoised.ply"]
ply_file = glob.glob(os.path.join(rootpath, 'seg_*.ply'))
threshold = 0.3  # Adjust this value based on your data

# Load point clouds
point_clouds = load_point_clouds(ply_file)

# Group overlapping point clouds
groups = group_overlapping_point_clouds(point_clouds, threshold)

# Print results
for i, group in enumerate(groups):
    print(f"Group {i + 1}: {[ply_file[j] for j in group]}")


# Visualize the results (optional)
def visualize_groups(point_clouds, groups):
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [0.5, 0, 0],  # Dark Red
        [0, 0.5, 0],  # Dark Green
        [0, 0, 0.5],  # Dark Blue
        [0.5, 0.5, 0],  # Olive
        [0.5, 0, 0.5],  # Purple
        [0, 0.5, 0.5],  # Teal
        [1, 0.5, 0],  # Orange
        [0, 1, 0.5],  # Spring Green
        [0.5, 0, 1],  # Violet
        [1, 0.5, 0.5],  # Light Coral
        [0.5, 1, 0.5],  # Pale Green
        [0.5, 0.5, 1],  # Light Blue
        [0.75, 0.75, 0],  # Gold
        [0.75, 0, 0.75]  # Orchid
    ]
    for i, group in enumerate(groups):
        color = colors[i % len(colors)]
        for j in group:
            point_clouds[j].paint_uniform_color(color)
    o3d.visualization.draw_geometries(point_clouds)

def save_group_pcd(groups):
    for i, group in enumerate(groups):
        file_list = [ply_file[j] for j in group]
        pcd_list = [o3d.io.read_point_cloud(file) for file in file_list]

        # Combine all point clouds
        combined_pcd = pcd_list[0]  # Start with the first point cloud
        for pcd in pcd_list[1:]:
            combined_pcd += pcd  # Append subsequent point clouds

        # Save the combined point cloud to a new .ply file
        grouped_name = file_list[0].rsplit('_', 2)[1]
        print("Saved the group: ", grouped_name)
        o3d.io.write_point_cloud(f'716/group/{grouped_name}_{i}.ply', combined_pcd)

visualize_groups(point_clouds, groups)
save_group_pcd(groups)