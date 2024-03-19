from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re

# RATTAN - File in ChartQA dataset
torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

# model_name = "ahmed-masry/unichart-chartqa-960"
model_name = "ahmed-masry/unichart-base-960"
# image_path = "/content/chart_example_1.png"
image_path = "chart_example_1.png"
# image_path = "../VisText-Dataset/images/10.png"
# input_prompt = "<chartqa> What is the lowest value in blue bar? <s_answer>"
input_prompt = "<extract_data_table> <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model.to(device)

image = Image.open(image_path).convert("RGB")
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids

print("Decode Done")

pixel_values = processor(image, return_tensors="pt").pixel_values

print("Pixel Values Done")

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

print("Output Done")

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.split("<s_answer>")[1].strip()
print(sequence)
