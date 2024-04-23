# Script to parse Vistext data to UniChart format
# FILE TO CHANGE VISTEXT TARGETS TO UNICHART FORMAT
import os
import json

# Read json file contents
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

# Read json file contents
# json_file = read_json_file('../VisText-Dataset/mmfact_vistext_annotation/mmfact_vistext_test.json')
json_file = read_json_file('./data/vistext/mmfact_vistext_test.json')
json_data = json.loads(json_file)

# Read csv and for each image id in dataframe, get corresponding text from json and override target column to new text from json file
import pandas as pd

input_csv_file = "./finetuned_unichart_on_unichart_pretrain_mask_2_run_on_unichart_pretrain.csv"
output_csv_file = "./finetuned_unichart_on_unichart_pretrain_mask_2_run_on_unichart_pretrain_updated.csv"

# Read csv file
df = pd.read_csv(input_csv_file)


def find_table_by_img_id(data, img_id):
    
    return "Table not found for img_id {}".format(img_id)

# For each image id in dataframe, get corresponding text from json and override target column to new text from json file
for index, row in df.iterrows():
    image_id = row['image_id']

    flag = 0
    for entry in json_data:
        if int(entry["img_id"]) == image_id:
            table = entry["table"]
            df.at[index, 'target'] = table
            flag = 1
            break
    
    if flag == 1:
        continue
    else:
        print("Table not found for img_id {}".format(image_id))

# Save updated dataframe to new csv file
df.to_csv(output_csv_file, index=False)