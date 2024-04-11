import os
import json
import pandas as pd
import shutil

def process_directory(base_path, directory):
    # Construct full paths to the images and tables directories
    images_dir = os.path.join(base_path, directory, 'png')
    tables_dir = os.path.join(base_path, directory, 'tables')
    
    # Dictionary to hold the mapping from image ID to data table
    output_list = []
    
    # List all files in the images directory
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.png'):
            # Get the base filename without the extension to match with the CSV
            img_id, _ = os.path.splitext(image_file)
            
            # Corresponding CSV file path
            csv_file_path = os.path.join(tables_dir, f'{img_id}.csv')
            
            # If the corresponding CSV exists, read it and add to the dataset
            if os.path.exists(csv_file_path):

                with open(csv_file_path, 'r') as file:
                    table_data = file.read()
                    table_data = table_data.rstrip('\n')
                    table_data = table_data.replace("\n", " & ").replace(",", " | ")

                output_list.append({"img_id": img_id, "table": table_data})
    
    # Save the dataset to a JSON file
    output_file = os.path.join(base_path, directory, f'chartqa_dte_{os.path.basename(directory)}.json')
    with open(output_file, 'w') as f:
        json.dump(output_list, f, indent=4)
        
    print(f'Processed {directory}, output file: {output_file}')

def process_data(base_path, directories):
    for directory in directories:
        process_directory(base_path, directory)

def copy_images_to_new_directory(base_path, directories):
    new_images_dir = os.path.join(base_path, 'images')
    
    # Create the new directory if it doesn't already exist
    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)
    
    for subdir in directories:
        png_dir_path = os.path.join(base_path, subdir, 'png')
        
        # Check if the png directory exists
        if os.path.exists(png_dir_path):
            # List all png files in the directory
            for image_file in os.listdir(png_dir_path):
                if image_file.endswith('.png'):
                    source_path = os.path.join(png_dir_path, image_file)
                    dest_path = os.path.join(new_images_dir, image_file)
                    
                    # Copy the image to the new directory, overwriting if necessary
                    shutil.copy(source_path, dest_path)
    
    print(f"All images have been copied to {new_images_dir}")

# Main function
if __name__ == '__main__':
    base_path = "../../datasets/ChartQA/ChartQADataset"
    # base_path = "../data/chartqa"
    directories = ['train', 'val', 'test']
    process_data(base_path, directories)
    # copy_images_to_new_directory(base_path, directories)



