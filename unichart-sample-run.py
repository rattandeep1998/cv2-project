from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
import matplotlib.pyplot as plt
import numpy as np

# Hide random patches in the image
def hide_patches(image, grid_size=(16, 16)):
    # Calculate average pixel value across the dataset (assuming a fixed value for simplicity)
    average_pixel_value = 128  # Placeholder: Replace with the actual dataset's average pixel value

    img_array = np.array(image)
    h, w, _ = img_array.shape
    patch_h, patch_w = h // grid_size[0], w // grid_size[1]

    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            if np.random.rand() < 0.5:  # Randomly choose to hide this patch
                img_array[i:i+patch_h, j:j+patch_w, :] = average_pixel_value
    
    return Image.fromarray(img_array)

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
hidden_image = hide_patches(image)

# Plot original and hidden images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(hidden_image)
ax[1].set_title("Image with Hidden Patches")
ax[1].axis('off')

plt.show()

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
