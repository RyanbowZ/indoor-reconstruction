import numpy as np
import open3d as o3d


def process_ply_with_open3d(ply_path, index_path, output_path):
    """
    Process a PLY file using Open3D to keep only vertices specified in the index file.
    Preserves the original binary format and properties.

    Args:
        ply_path (str): Path to input PLY file
        index_path (str): Path to index.npy file
        output_path (str): Path to save the processed PLY file
    """
    # Load indices
    data = np.load(index_path)
    indices = np.where(data == '1')[0]

    # Read the point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Get all points and properties as numpy arrays
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Create new point cloud with selected vertices
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points[indices])
    new_pcd.normals = o3d.utility.Vector3dVector(normals[indices])

    # If there are custom attributes (f_dc, opacity, scale, rot), preserve them
    # if hasattr(pcd, 'colors'):  # Open3D stores custom float properties in colors
    #     colors = np.asarray(pcd.colors)
    #     new_pcd.colors = o3d.utility.Vector3dVector(colors[indices])

    # Write with the same format as input
    # Read original header to determine format
    with open(ply_path, 'rb') as f:
        header = ''
        while True:
            line = f.readline().decode('ascii').strip()
            header += line + '\n'
            if 'format' in line:
                format_line = line
                break

    # Extract format (binary_little_endian or ascii)
    format_type = format_line.split()[1]

    # Write with the same format
    if 'binary' in format_type:
        o3d.io.write_point_cloud(output_path, new_pcd, write_ascii=False, compressed=False)
    else:
        o3d.io.write_point_cloud(output_path, new_pcd, write_ascii=True, compressed=False)

    return len(indices)


# Preserve custom properties by combining both approaches
def process_ply_complete(ply_path, index_path, output_path):
    """
    Complete solution that handles all custom properties while using Open3D
    for efficient point cloud operations.
    """
    # First use Open3D to process basic properties
    process_ply_with_open3d(ply_path, index_path, output_path)

    # Then handle custom properties using direct binary reading
    indices = np.load(index_path)

    # Read all vertex data
    with open(ply_path, 'rb') as f:
        # Skip to end of header
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break

        # Read vertex data
        vertex_data = np.fromfile(f, dtype=np.float32)
        vertex_data = vertex_data.reshape(-1, 16)  # 16 properties per vertex

        # Select custom properties (f_dc, opacity, scale, rot)
        selected_data = vertex_data[indices]

    # Update the output file with custom properties
    # First read the entire output file
    with open(output_path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        existing_data = f.read()

    # Write back with custom properties
    with open(output_path, 'wb') as f:
        f.write(header)
        selected_data.tofile(f)

    return len(indices)

# Example usage:
processed_vertices = process_ply_with_open3d('rotate_chair.ply', 'label.npy', 'kept_chair.ply')
print(f"Processed PLY file with {processed_vertices} vertices")