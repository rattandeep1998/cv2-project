import pandas as pd
import json

# df = pd.read_csv("../data/chartqa/train/unichart_formatted_chart_type_annotations.csv")

# # Count the number of charts of each type
# chart_type_counts = df['type'].value_counts()

# print(chart_type_counts)

# Vistext Chart Type
json_path = "../data/vistext/mmfact_vistext_test.json"
chart_type_count = {}

with open(json_path, 'r') as f:
    data_list = json.load(f)

    for data in data_list:
        chart_type = data['chart_type']
        if chart_type in chart_type_count:
            chart_type_count[chart_type] += 1
        else:
            chart_type_count[chart_type] = 1

print(chart_type_count)