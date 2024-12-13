import open3d as o3d
import numpy as np
import os
import json


def compute_centroid_and_bbox(point_cloud):
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    bbox = point_cloud.get_axis_aligned_bounding_box()
    scale = bbox.get_extent()
    return centroid, scale, bbox


def visualize_centroids_and_bboxes(base_path, pcl_dir, output_json):
    # Load the base point cloud
    base_pcl = o3d.io.read_point_cloud(base_path)
    temp_pcl = o3d.io.read_triangle_mesh("floor_plan_grid.obj")
    # Initialize lists for centroids and bounding boxes
    centroids = []
    bounding_boxes = []
    json_data = []
    # Load and process each point cloud
    for file in os.listdir(pcl_dir):
        if file.endswith('.ply'):
            pcl_path = os.path.join(pcl_dir, file)
            point_cloud = o3d.io.read_point_cloud(pcl_path)
            centroid, scale, bbox = compute_centroid_and_bbox(point_cloud)
            label_name = file.rsplit('_', 1)[0]
            label_index = file.rsplit('_', 1)[1].rsplit('.', 1)[0]
            if file.startswith('table'):
                bbox.color = [0, 0, 1]  # Green color for bounding boxes
                wordnet = "04379243"
            else:
                bbox.color = [0, 1, 0]  # Green color for bounding boxes
                wordnet = "03001627"
            print(file, centroid)
            centroids.append(centroid)
            bounding_boxes.append(bbox)
            json_data.append({
                "file": file,
                "label": label_name,
                "index": label_index,
                "wordnet": wordnet,
                "centroid": {
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2])
                },
                "scale": {
                    "x": float(scale[0]),  # Width
                    "y": float(scale[1]),  # Height
                    "z": float(scale[2])  # Depth
                }
            })

    # Prepare visualization
    vis_objects = [base_pcl, temp_pcl]

    # Add centroids as spheres
    for centroid in centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1, 0, 0])  # Red color for centroids
        vis_objects.append(sphere)

    # Add bounding boxes
    for bbox in bounding_boxes:

        vis_objects.append(bbox)

    # Visualize
    o3d.visualization.draw_geometries(vis_objects)
    with open(output_json, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# Usage
base_ply_path = 'scan716_aligned.ply'
point_cloud_dir = '716/group/denoise'
output_json = '716/scene_denoise.json'
visualize_centroids_and_bboxes(base_ply_path, point_cloud_dir, output_json)
