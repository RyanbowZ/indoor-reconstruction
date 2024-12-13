import open3d as o3d
import numpy as np

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# Load the point cloud
print("Loading point cloud...")
source_path = "./716/group/"
pcd = o3d.io.read_point_cloud(f"{source_path}table_11.ply") # /seg_object_on_slam/7/pts.ply
print(f"Points before denoising: {len(pcd.points)}")

# Visualization of input
print("Input point cloud:")
# o3d.visualization.draw_geometries([pcd])

# Method 1: Statistical outlier removal
print("Statistical outlier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
display_inlier_outlier(pcd, ind)

# Method 2: Radius outlier removal
print("Radius outlier removal")
cl, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.4)  # 40 0.3; 50 0.4; Other: 100, 0.5; 1st chair: 160, 0.6
display_inlier_outlier(pcd, ind)

# Choose the method you prefer and apply it
print("Applying selected denoising method...")
# 1st round setting pcd_denoised, _ = pcd.remove_radius_outlier(nb_points=50, radius=0.4)
# pcd_denoised, _ = pcd_denoised.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)

pcd_denoised, _ = pcd.remove_radius_outlier(nb_points=50, radius=0.4)
pcd_denoised, _ = pcd_denoised.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)


print(f"Points after denoising: {len(pcd_denoised.points)}")
o3d.visualization.draw_geometries([pcd_denoised])
#
# Save the denoised point cloud
o3d.io.write_point_cloud(f"{source_path}table_11_denoised.ply", pcd_denoised)
print("Denoised point cloud saved")