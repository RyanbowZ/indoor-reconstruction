import os
import shutil
import numpy as np

# Define the root folder path
root_folder = './seg_object_on_slam/'
threshold = 80
# Iterate over all items in the root folder
for subdir, dirs, files in os.walk(root_folder):
    # Check if the current folder contains a .ply file
    if any(file.endswith('.ply') for file in files):
        # Iterate through the files again to find .npy files
        for file in files:
            if file.endswith('.npy'):
                # Get the full path of the .npy file
                subfolder_name = os.path.basename(subdir)
                npy_file_path = os.path.join(subdir, file)
                npy_data = np.load(npy_file_path)
                count_ones = np.count_nonzero(npy_data == 1)
                if count_ones > threshold:
                # Define the destination path (root folder 'a')
                    destination_path = os.path.join(root_folder, subfolder_name+"_"+file)
                    # Copy the .npy file to the root folder
                    shutil.copy(npy_file_path, destination_path)
                    print(f"Copied {npy_file_path} to {destination_path}")
