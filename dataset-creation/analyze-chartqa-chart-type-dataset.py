import pandas as pd

df = pd.read_csv("../data/chartqa/train/unichart_formatted_chart_type_annotations.csv")

# Count the number of charts of each type
chart_type_counts = df['type'].value_counts()

print(chart_type_counts)