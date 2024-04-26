from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# List of model names
model_names = [
    # "ahmed-masry/unichart-base-960",
    "../content/output_data/unichart-on-pretrain-unichart-dte-no-mask-2/chartqa-checkpoint-epoch=2-30942",
    # "../content/output_data/unichart-on-pretrain-unichart-dte-mask-2/chartqa-checkpoint-epoch=2-34380",
]

IMAGE_URL = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png'
# image_path = 'chart_example_1.png'
# image_path = '../data/chartqa/train/png/12044.png'
# image_path = '../data/chartqa/train/png/two_col_104783.png'
# image_path = '../data/vistext/images/5.png'
image_path = '../data/chartqa/train/png/85839291000279.png'
# 34 and 63 and 288 and 7059
# image_path = '../data/vistext/images/7059.png'

input_prompt = "<extract_data_table> <s_answer>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def hide_patches(image, grid_size=(16, 16)):
    average_pixel_value = 127.5
    img_array = np.array(image)
    h, w, _ = img_array.shape
    patch_h, patch_w = h // grid_size[0], w // grid_size[1]

    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            if np.random.rand() < 0.2:
                img_array[i:i+patch_h, j:j+patch_w, :] = average_pixel_value
    return Image.fromarray(img_array)

def inference(model, processor, image, prompt):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_attentions=True
    )
    return outputs

def create_heatmap(image, outputs, ax, title):
    attention = outputs.encoder_attentions[0][0, 0]
    att_map = torch.nn.functional.interpolate(
        attention.unsqueeze(0).unsqueeze(0), 
        size=image.size[::-1], 
        mode='bilinear',
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    ax.imshow(image)
    ax.imshow(att_map, cmap='jet', alpha=0.6)
    ax.axis('off')
    ax.set_title(title)

def print_sequence(processor, outputs, isHidden):
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.split("<s_answer>")[1].strip()

    if isHidden:
        print(f"Hidden Sequence: {sequence}")
    else:
        print(f"Non-Hidden Sequence: {sequence}")

def run_model(model_name, image_path, input_prompt):
    print(f"Processing model: {model_name}")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = DonutProcessor.from_pretrained(model_name)
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    hidden_image = hide_patches(image)

    outputs = inference(model, processor, image, input_prompt)
    outputs_hidden = inference(model, processor, hidden_image, input_prompt)

    print_sequence(processor, outputs, False)
    print_sequence(processor, outputs_hidden, True)

    return image, hidden_image, outputs, outputs_hidden

def main():
    # fig, axs = plt.subplots(len(model_names), 2, figsize=(20, 15), squeeze=False)  # Adjust as needed

    # for i, model_name in enumerate(model_names):
    #     image, hidden_image, outputs, outputs_hidden = run_model(model_name, image_path, input_prompt)

    #     create_heatmap(image, outputs, axs[i, 0], f"Original")
    #     create_heatmap(hidden_image, outputs_hidden, axs[i, 1], f"Hidden")

    # plt.tight_layout()
    # plt.savefig("combined_plots_one_vistext_7059.png")
    # plt.close()

    image = Image.open(image_path).convert("RGB")
    hidden_image = hide_patches(image)
    image.save("original_image.png")
    
    # Save hidden image
    hidden_image.save("hidden_image.png")

if __name__ == "__main__":
    main()
