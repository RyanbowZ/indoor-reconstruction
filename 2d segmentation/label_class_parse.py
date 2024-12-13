import os
import json


def extract_instance_classes(directory):
    # Initialize an empty dictionary to store instance_id: class_name pairs
    instance_classes = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("mask_") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            # Open and load the JSON data
            with open(filepath, 'r') as file:
                data = json.load(file)

                # Process each label in the JSON data
                for label in data.get("labels", {}).values():
                    instance_id = label.get("instance_id")
                    class_name = label.get("class_name")

                    # Only add to the dictionary if the instance_id is not already present
                    if instance_id not in instance_classes and class_name:
                        instance_classes[instance_id] = class_name

    return instance_classes


# Directory containing the mask_xxx.json files
directory_path = './outputs/rendered_716_new/json_data'

# Call the function and print the result
first_instance_classes = extract_instance_classes(directory_path)
print(first_instance_classes)

with open(f'{directory_path}/label.json', 'w') as file:
    json.dump(first_instance_classes, file, indent=4)