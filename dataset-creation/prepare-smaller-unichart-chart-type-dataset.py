import json
import random

def reduce_json_file_size(original_file_path, new_file_path, reduction_factor=0.01):
    # Read the original JSON data
    with open(original_file_path, 'r') as file:
        data = json.load(file)
    
    # Determine the new size of the data (10% of the original size)
    new_size = int(len(data) * reduction_factor)
    
    print(f"{new_size}")

    # Randomly select a subset of the data
    reduced_data = random.sample(data, new_size)
    
    # Write the reduced data to the new file path
    with open(new_file_path, 'w') as file:
        json.dump(reduced_data, file, indent=4)

# File paths
train_file_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_train.json'
val_file_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_val.json'
test_file_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_test.json'

new_train_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_small_train.json'
new_val_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_small_val.json'
new_test_path = '../data/unichart-pretrain/updated_unichart_pretrain_datatable_small_test.json'

# Reduce the size of each file and save at new locations
reduce_json_file_size(train_file_path, new_train_path)
reduce_json_file_size(val_file_path, new_val_path)
reduce_json_file_size(test_file_path, new_test_path)
