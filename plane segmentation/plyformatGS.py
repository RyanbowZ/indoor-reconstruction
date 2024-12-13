import struct
import os

def convert_ply_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'rb') as f:
        # Read and parse the header
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break

        # Extract the number of vertices
        num_vertices = int([line for line in header if line.startswith('element vertex')][0].split()[-1])

        # Read the binary data
        vertex_size = struct.calcsize('17f')  # 17 float properties in the input format
        data = f.read(num_vertices * vertex_size)

    # Write the output file
    with open(output_file, 'wb') as f:
        # Write the new header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_vertices}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
        f.write(b"property float f_dc_0\n")
        f.write(b"property float f_dc_1\n")
        f.write(b"property float f_dc_2\n")
        for i in range(45):
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\n")
        f.write(b"property float scale_1\n")
        f.write(b"property float scale_2\n")
        f.write(b"property float rot_0\n")
        f.write(b"property float rot_1\n")
        f.write(b"property float rot_2\n")
        f.write(b"property float rot_3\n")
        f.write(b"end_header\n")

        # Process and write the vertex data
        for i in range(num_vertices):
            vertex = struct.unpack('17f', data[i*vertex_size:(i+1)*vertex_size])
            new_vertex = vertex[:9] + (0,) * 45 + vertex[9:]  # Add 45 zeros for f_rest_0 to f_rest_44
            f.write(struct.pack('62f', *new_vertex))

    print(f"Conversion complete. Output file: {output_file}")

# Usage
input_file = './716/splat.ply'
output_file = './716/splat_formatted.ply'
convert_ply_file(input_file, output_file)