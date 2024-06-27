import os

# Function to rename images
def rename_images(dataset_dir):
    print('Started Renaming...')

    for root, dirs, files in os.walk(dataset_dir):
        count = 0
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                # Extract bias category and image class from the path
                parts = root.split(os.sep)
                bias_category = parts[-3]  # e.g., age, gender
                subcategory = parts[-2]  # e.g., middle-aged, male
                image_class = parts[-1]  # e.g., angry, happy

                # Get the original file extension
                file_extension = os.path.splitext(file)[1]

                # Generate the new file name
                new_file_name = f"{bias_category}-{image_class}-{subcategory}_{count:04d}{file_extension}"
                new_file_path = os.path.join(root, new_file_name)

                # Rename the file
                os.rename(image_path, new_file_path)

                count += 1

    print('Completed Renaming.')

# Define the dataset directory and call the function
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dataset_dir = os.path.join(parent_dir, 'dataset-bias_attributes')  # Finds 'dataset-bias_attributes' in the parent directory

rename_images(dataset_dir)
