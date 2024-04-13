import pandas as pd
import json
import re
from collections import Counter

def count_tasks_in_queries(dataframe):
    # Regex to find patterns in the form of <task_name>
    pattern = re.compile(r'<(.*?)>')
    
    # List to store all found task names
    all_tasks = []
    
    # Iterate over each label in the DataFrame
    for label in dataframe['query']:
        # Find all occurrences of the pattern
        tasks = pattern.findall(label)
        all_tasks.extend(tasks)
    
    # Count occurrences of each task
    task_counts = Counter(all_tasks)
    
    return task_counts

# Load the three parquet files into a DataFrame
base_path_unichart = '../data/unichart-pretrain/unichart-pretrain-data/data/'
df = pd.concat([
    pd.read_parquet(base_path_unichart + 'train-00000-of-00003-db40b2e51df9cb23.parquet'),
    pd.read_parquet(base_path_unichart + 'train-00001-of-00003-176f88b6a51ec36d.parquet'),
    pd.read_parquet(base_path_unichart + 'train-00002-of-00003-1e538839dce74b46.parquet')
])

print(len(df))

filtered_df = df[df['query'].str.startswith("<extract_data_table>")]

print(filtered_df.head())

formatted_data = [{
    "img_id": row["imgname"][:-4] if row["imgname"].endswith('.png') else row["imgname"],
    "table": row["label"]
} for index, row in filtered_df.iterrows()]

# task_counts = count_tasks_in_queries(df)
# print(task_counts)

with open('../data/unichart-pretrain/filtered_unichart_pretrain_datatable.json', "w") as file:
    json.dump(formatted_data, file, indent=4)