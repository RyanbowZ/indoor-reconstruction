import numpy as np
import struct


def process_ply_with_index(ply_path, index_path, output_path):
    """
    Process a PLY file to keep only vertices specified in the index file.

    Args:
        ply_path (str): Path to input PLY file
        index_path (str): Path to index.npy file
        output_path (str): Path to save the processed PLY file
    """
    # Load indices
    data = np.load(index_path)
    indices = np.where(data == '1')[0]

    # Read the header from the original PLY file
    header_lines = []
    with open(ply_path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

    # Parse number of vertices from header
    for line in header_lines:
        if 'element vertex' in line:
            num_vertices = int(line.split()[-1])
            break

    # Calculate size of each vertex in bytes
    # 16 float properties * 4 bytes per float
    vertex_size = 16 * 4

    # Read all vertices
    with open(ply_path, 'rb') as f:
        # Skip header
        for line in header_lines:
            f.readline()

        # Read all vertex data
        vertex_data = f.read(num_vertices * vertex_size)

    # Convert binary data to numpy array
    vertices = np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 16)

    # Select only the vertices we want to keep
    selected_vertices = vertices[indices]

    # Write new PLY file
    with open(output_path, 'wb') as f:
        # Update header with new vertex count
        new_header = header_lines.copy()
        for i, line in enumerate(new_header):
            if 'element vertex' in line:
                new_header[i] = f'element vertex {len(indices)}'

        # Write header
        f.write('\n'.join(new_header).encode('ascii'))
        f.write(b'\n')

        # Write selected vertices
        selected_vertices.tofile(f)

    return len(indices)

# Example usage:
processed_vertices = process_ply_with_index('rotate_chair.ply', 'label.npy', 'kept_chair.ply')
print(f"Processed PLY file with {processed_vertices} vertices")