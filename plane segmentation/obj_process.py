import open3d as o3d
import numpy as np

def align_object_to_plane(obj, plane_normal):
    # Normalize the plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # Calculate the rotation axis (cross product of y-axis and plane normal)
    rotation_axis = np.cross([0, 1, 0], plane_normal)
    # Calculate the rotation angle
    rotation_angle = np.arccos(np.dot([0, 1, 0], plane_normal))
    # Create the rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    # Apply the rotation to the object
    obj.rotate(rotation_matrix, center=(0, 0, 0))

    return obj

# Load the OBJ file
mesh = o3d.io.read_point_cloud("39-cup/Cup.ply")

# Get the bounding box of the mesh
bbox = mesh.get_axis_aligned_bounding_box()
half_extents = bbox.get_half_extent()

# Find the longest axis of the bounding box
longest_axis = max(half_extents)

# Compute the scaling factor to make the longest axis 1 unit
scale = 0.1 / (2.0 * longest_axis)

# Scale and center the mesh
mesh.scale(scale, center=mesh.get_center())
mesh.translate(-mesh.get_center())  # Move the mesh to the origin

align_object_to_plane(mesh, [0.4, 0.3, 0.7])

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the scaled and centered mesh to the visualization
vis.add_geometry(mesh)
#vis.add_geometry(rotated_mesh)

# Set up the camera view
vis.get_view_control().set_front([0, 0, -1])
vis.get_view_control().set_lookat(np.zeros(3))
vis.get_view_control().set_up([0, 1, 0])

# Run the visualization
vis.run()
vis.destroy_window()

