import json
import random

def split_dataset(data, train_percent, val_percent, test_percent, output_path_prefix):
    total_entries = len(data)

    # characteristic_count = sum(1 for entry in data if entry['table'].startswith('Years |'))
    # print(f"Characteristic count: {characteristic_count}")
    
    filtered_data = [entry for entry in data if not entry['table'].startswith('Characteristic')]
    filtered_entries = len(filtered_data)

    print(f"Total entries: {total_entries}")
    print(f"Entries after Filter: {filtered_entries}")

    total_entries = filtered_entries
    data = filtered_data

    train_size = int(total_entries * train_percent / 100)
    val_size = int(total_entries * val_percent / 100)
    test_size = int(total_entries * test_percent / 100)

    random.shuffle(data)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size: train_size + val_size + test_size]

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    with open(f'{output_path_prefix}_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(f'{output_path_prefix}_val.json', 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(f'{output_path_prefix}_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

file_path = '../data/unichart-pretrain/filtered_unichart_pretrain_datatable.json'
output_path_prefix = '../data/unichart-pretrain/filtered_unichart_pretrain_datatable'

# file_path = '../../datasets/unichart-pretrain-data/filtered_unichart_pretrain_datatable.json'
# output_path_prefix = '../../datasets/unichart-pretrain-data/filtered_unichart_pretrain_datatable'

# Load data from the JSON file
with open(file_path, 'r') as file:
    formatted_data = json.load(file)

split_dataset(formatted_data, train_percent=60, val_percent=2, test_percent=2, output_path_prefix=output_path_prefix)
