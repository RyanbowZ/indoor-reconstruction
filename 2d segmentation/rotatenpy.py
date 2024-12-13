import os
from PIL import Image
import numpy as np



def convert_images(folder_path):
    # Ensure the output directory exists
    output_folder = os.path.join(folder_path, "rotatenpy")
    os.makedirs(output_folder, exist_ok=True)

    count = 1
    for filename in os.listdir(folder_path):
        # Check if the file is a PNG image
        if filename.lower().endswith('.npy'):
            npy_path = os.path.join(folder_path, filename)
            data = np.load(npy_path)
            rotated_data = np.rot90(data)
            output_path = os.path.join(output_folder, filename)
            np.save(output_path, rotated_data)
            # Open the image, rotate, and convert to JPG
            # with Image.open(img_path) as img:
            #     rotated_img = img.rotate(-90, expand=True)  # Rotate 90 degrees clockwise
            #     # Generate new filename
            #     new_filename = f"{str(count).zfill(4)}.jpg"
            #     output_path = os.path.join(output_folder, new_filename)
            #
            #     # Save as JPEG
            #     rotated_img.convert("RGB").save(output_path, "JPEG")

            print(f"Converted and saved: {output_path}")
            count += 1


# Specify the path to the folder containing PNG images
folder_path = "outputs/414/mask_data"
convert_images(folder_path)
