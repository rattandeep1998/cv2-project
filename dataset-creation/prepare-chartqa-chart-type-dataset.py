import os
import json
import pandas as pd

def process_directory(directory_path):
    # Initialize lists to hold the data
    image_ids = []
    chart_types = []
    
    # Loop through all JSON files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Extract image id (filename without extension) and chart type
                image_id = filename[:-5]  # Remove .json
                chart_type = data['type']
                # Append to lists
                image_ids.append(image_id)
                chart_types.append(chart_type)
    
    # Trim the dataset
    image_ids = image_ids[:10]
    chart_types = chart_types[:10]

    # Create DataFrame
    df = pd.DataFrame({
        'id': image_ids,
        'type': chart_types
    })
    
    # Save DataFrame to CSV in the same main directory as the JSON files
    csv_path = os.path.join(os.path.dirname(directory_path), 'unichart_formatted_chart_type_' + os.path.basename(directory_path) + '_small.csv')
    df.to_csv(csv_path, index=False)

# Base path for the dataset
base_path = '../data/chartqa'

# Process each dataset partition
for partition in ['train', 'val', 'test']:
    partition_path = os.path.join(base_path, partition)
    directory_path = os.path.join(base_path, partition, 'annotations')
    process_directory(directory_path)