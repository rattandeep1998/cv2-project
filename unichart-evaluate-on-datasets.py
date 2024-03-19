import pandas as pd
import deplot_metrics as deplot_metrics
import json

dataset = "chartqa"

# Update this path
csv_file_path = "unichart_on_" + dataset + ".csv"

df = pd.read_csv(csv_file_path)

# Print the first 5 rows of the dataframe
print(df.head())

# Extract targets and predictions
targets = df['target'].tolist()
predictions = df['prediction'].tolist()

# Calculate metrics
metric = {}
metric.update(deplot_metrics.table_datapoints_precision_recall(targets, predictions))
metric.update(deplot_metrics.table_number_accuracy(targets, predictions))
metric_log = json.dumps(metric, indent=2)

print(metric_log)