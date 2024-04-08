from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
from tqdm import tqdm
import pandas as pd
import json

# UniChart model
# model_name = "ahmed-masry/unichart-base-960"
model_name = "/content/output_data/chartqa-checkpoint-epoch=1-8000"

input_prompt = "<extract_data_table> <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Device: ", device)

dataset = "vistext"

# DATA PRE_PROCESSING
if dataset == "chartqa":
    image_base_path = "./data/chartqa/test/png/"
elif dataset == "vistext":
    image_base_path = "./data/vistext/images/"

# LOAD IMAGES

# RATTAN - Does not handle 2 column and multi-column right now

images = {}

tot_count = 0

# Read 10 images from image_base_path directory and store that value in image dictionary
for file in os.listdir(image_base_path):
    tot_count += 1
    if re.search(r'\.png$', file):
        image = Image.open(image_base_path + file).convert("RGB")
        # File name should be without extension
        file = file.split('.')[0]
        images[file] = image

    # TODO - Remove this condition later
    # if tot_count == 100:
    #     break

print("Total Images Read: ", len(images))

# LOAD TARGET DATATABLES

# Initialize an empty list to store data
targets_dictionary = {}

if dataset == "chartqa":
    csv_directory_path = "./data/chartqa/test/tables/"
    csv_files = [file for file in os.listdir(csv_directory_path) if file.endswith('.csv')]

    # Read each CSV file, convert its content to a single string, and append to the list
    for file in csv_files:
        file_path = os.path.join(csv_directory_path, file)
        with open(file_path, 'r') as f:
            content = f.read()
            # Removing the '.csv' extension and using the filename as image_id
            image_id = file.replace('.csv', '')
            targets_dictionary[image_id] = content
elif dataset == "vistext":
    # Running only on vistext TEST data
    datatable_directory_path = "./data/vistext/data_test.json"

    with open(datatable_directory_path) as f:
        data = json.load(f)
        df = pd.DataFrame(data)
    
    # Traverse through dataframe and store the image_id and table in targets_dictionary
    for index, row in df.iterrows():
        image_id = row['img_id']
        datatable = row['datatable']
        targets_dictionary[image_id] = datatable

print("Total Data Points Read: ", len(targets_dictionary))

# RUN THE MODEL

print("Starting Model Predictions")

predictions_dictionary = {}

# Run the model on the images
# Loop over images and generate the output
for i, image in tqdm(images.items()):

    # Added check for VisText dataset
    if i in targets_dictionary.keys():
        
        decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.split("<s_answer>")[1].strip()

        predictions_dictionary[i] = sequence

print("Total Predictions: ", len(predictions_dictionary))

# Storing the model outputs
results_output_path = "unichart_on_finetuned_" + dataset + ".csv"

# Map the predictions to target data and save in dataframe with 3 columns image_id, prediction, target
dataframe_rows = []
ignored_data_points = 0

for image_id in predictions_dictionary.keys():
    if image_id in targets_dictionary:
        row = {'image_id': image_id, 'prediction': predictions_dictionary[image_id], 'target': targets_dictionary[image_id]}
        dataframe_rows.append(row)
    else:
        ignored_data_points += 1

df = pd.DataFrame(dataframe_rows)
df.to_csv(results_output_path, index=False)

print("Total Written Rows: ", len(df))
print("Ignored Data Points: ", ignored_data_points)
