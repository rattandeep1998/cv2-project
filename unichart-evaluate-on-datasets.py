import pandas as pd
import deplot_metrics as deplot_metrics
import json


def preprocess_outputs(targets, predictions, dataset):
    if dataset == "chartqa":
        # Replace & with <0x0a> and individual strings to array
        predictions = [prediction.replace("&", "<0x0a>") for prediction in predictions]

        # Replace new line with <0x0a> and , with ' | '
        targets = [target.replace("\n", "<0x0a>").replace(",", " | ") for target in targets]
        targets = [[target] for target in targets]

        return targets, predictions
    elif dataset == "vistext":
        # Replace & with <0x0a> and individual strings to array
        predictions = [prediction.replace("&", "<0x0a>") for prediction in predictions]

        targets = [target.replace("&", "<0x0a>") for target in targets]
        targets = [[target] for target in targets]

        return targets, predictions


# dataset = "chartqa"
# csv_file_path = "unichart_on_chartqa_backup.csv"
# processed_csv_file_path = "unichart_on_chartqa_processed.csv"

# dataset = "chartqa"
# csv_file_path = "finetuned_unichart_on_unichart_pretrain_run_on_chartqa.csv"
# processed_csv_file_path = "finetuned_unichart_on_unichart_pretrain_run_on_chartqa_processed.csv"

dataset = "vistext"
# Since while running Unichart model, it is picking target datapoints directly from CSV,
# we do not need to change the dataset type here or do some different formatting.
csv_file_path = "unichart_run_on_unichart_pretrain.csv"
processed_csv_file_path = "unichart_run_on_unichart_pretrain_processed.csv"

# dataset = "vistext"
# csv_file_path = "finetuned_unichart_on_chartqa_formatted_run_on_vistext_updated.csv"
# processed_csv_file_path = "finetuned_unichart_on_chartqa_formatted_run_on_vistext_processed.csv"

# dataset = "vistext"
# csv_file_path = "unichart_on_finetuned_vistext_updated.csv"
# processed_csv_file_path = "unichart_on_finetuned_vistext_processed.csv"


df = pd.read_csv(csv_file_path)

# Extract targets and predictions
targets = df['target'].tolist()
predictions = df['prediction'].tolist()

targets, predictions = preprocess_outputs(targets, predictions, dataset)

# Output targets, predictions to csv
df['target'] = targets
df['prediction'] = predictions

df.to_csv(processed_csv_file_path, index=False)

# Print the first 5 rows of the dataframe
print(df.head())

print("Length of Targets: ", len(targets))
print("Length of Predictions: ", len(predictions))

# Calculate metrics
metric = {}
metric.update(deplot_metrics.table_datapoints_precision_recall(targets, predictions))
# metric.update(deplot_metrics.row_datapoints_precision_recall(targets, predictions))
metric.update(deplot_metrics.table_number_accuracy(targets, predictions))
metric_log = json.dumps(metric, indent=2)

print(metric_log)
