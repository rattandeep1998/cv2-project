from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# model_name = "ahmed-masry/unichart-chartqa-960"
model_name = "ahmed-masry/unichart-base-960"
IMAGE_URL = 'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png'
# image_path = 'chart_example_1.png'
# image_path = './data/chartqa/train/png/12044.png'
# image_path = './data/chartqa/train/png/two_col_104783.png'
image_path = './data/vistext/images/5.png'
input_prompt = "<extract_data_table> <s_answer>"
# input_prompt = "<chartqa> What is the lowest value in blue bar? <s_answer>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def print_sequence(processor, outputs, isHidden):
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.split("<s_answer>")[1].strip()

    if isHidden:
        print(f"Hidden Sequence: {sequence}")
    else:
        print(f"Non-Hidden Sequence: {sequence}")

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
        output_attentions=True,
    )
    
    # Check and print details about cross_attentions, if available
    # if hasattr(outputs, 'cross_attentions'):
    #     print("cross_attentions is available.")
    #     for i, layer in enumerate(outputs.cross_attentions):
    #         print(f"Layer {i}: Type={type(layer)}, Length={len(layer)}")
    #         if isinstance(layer, torch.Tensor):
    #             print(f"Tensor shape: {layer.shape}")
    #         elif isinstance(layer, tuple):
    #             print("Contents of tuple:")
    #             for j, tensor in enumerate(layer):
    #                 print(f"Tensor {j} shape: {tensor.shape}")
    # else:
    #     print("cross_attentions is not available.")
    
    return outputs

# Hide random patches in the image
def hide_patches(image, grid_size=(16, 16)):
    # Calculate average pixel value across the dataset (assuming a fixed value for simplicity)
    average_pixel_value = 127.5  # Placeholder: Replace with the actual dataset's average pixel value

    img_array = np.array(image)
    h, w, _ = img_array.shape
    patch_h, patch_w = h // grid_size[0], w // grid_size[1]

    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            if np.random.rand() < 0.3:  # Randomly choose to hide this patch
                img_array[i:i+patch_h, j:j+patch_w, :] = average_pixel_value
    
    return Image.fromarray(img_array)

def create_heatmap(image, outputs, title, file_path):
    attention = outputs.encoder_attentions[-1][0, 0]
    att_map = torch.nn.functional.interpolate(
        attention.unsqueeze(0).unsqueeze(0), 
        size=image.size[::-1], 
        mode='bilinear',
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(att_map, cmap='jet', alpha=0.6)
    ax.axis('off')
    ax.set_title(title)
    plt.savefig(file_path)
    plt.close()

def create_dual_heatmap(image1, outputs1, title1, image2, outputs2, title2, file_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Creates a figure with two subplots

    # Function to process each image and add to subplot
    def process_image(image, outputs, ax, title):
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

    # Process each image and place on respective subplot
    process_image(image1, outputs1, axes[0], title1)
    process_image(image2, outputs2, axes[1], title2)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_all_encoder_attentions(image, outputs, title_prefix, base_file_path):
    num_layers = len(outputs.encoder_attentions)
    num_columns = 4  # Set the number of columns in the subplot grid
    num_rows = (num_layers + num_columns - 1) // num_columns  # Calculate the required number of rows
    
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 5))
    fig.suptitle('Encoder Attention Across All Layers')

    axs = np.array(axs).reshape(num_rows, num_columns)

    for i, attention in enumerate(outputs.encoder_attentions):
        # print(f"Attention Shape: {attention.shape}")
        att_map = torch.nn.functional.interpolate(
            attention[0, 0].unsqueeze(0).unsqueeze(0), 
            size=image.size[::-1], 
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy()

        ax = axs[i // num_columns, i % num_columns]
        ax.imshow(image)
        ax.imshow(att_map, cmap='jet', alpha=0.6)
        ax.axis('off')
        ax.set_title(f'{title_prefix} Layer {i+1}')

    # Hide unused axes if any
    for ax in axs.flat[num_layers:]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(base_file_path)
    plt.close()

def plot_attention_maps(image, outputs, num_layers, title_prefix, base_file_path):
    fig, axs = plt.subplots(2, num_layers, figsize=(num_layers * 5, 10))  # 2 rows for encoder and cross-attentions
    fig.suptitle('Attention Maps Across Layers')

    for i in range(num_layers):
        # Encoder attentions
        encoder_attention = outputs.encoder_attentions[i]
        encoder_att_map = torch.nn.functional.interpolate(
            encoder_attention[0, 0].unsqueeze(0).unsqueeze(0),
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy()
        axs[0, i].imshow(image)
        axs[0, i].imshow(encoder_att_map, cmap='jet', alpha=0.6)
        axs[0, i].axis('off')
        axs[0, i].set_title(f'Encoder Layer {i+1}')

        # Cross-attentions, if available
        if hasattr(outputs, 'cross_attentions'):
            cross_attention = outputs.cross_attentions[i][1]
            # Accessing first tensor of tuple from each layer
            cross_att_map = torch.nn.functional.interpolate(
                cross_attention[0, 0].unsqueeze(0).unsqueeze(0), # Accessing first head of first batch
                size=image.size[::-1],
                mode='bilinear',
                align_corners=False
            ).squeeze().detach().cpu().numpy()
            axs[1, i].imshow(image)
            axs[1, i].imshow(cross_att_map, cmap='jet', alpha=0.6)
            axs[1, i].axis('off')
            axs[1, i].set_title(f'Cross Layer {i+1}')

    plt.tight_layout()
    plt.savefig(f"{base_file_path}.png")
    plt.close()

def main():
    # RATTAN - File in ChartQA dataset
    # torch.hub.download_url_to_file(IMAGE_URL, image_path)

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = DonutProcessor.from_pretrained(model_name)
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    hidden_image = hide_patches(image)

    outputs = inference(model, processor, image, input_prompt)
    outputs_hidden = inference(model, processor, hidden_image, input_prompt)

    print_sequence(processor, outputs, False)
    print_sequence(processor, outputs_hidden, True)

    # create_heatmap(image, outputs, "Original Image Attention", "original_attention.png")
    # create_heatmap(hidden_image, outputs_hidden, "Hidden Patches Image Attention", "hidden_attention.png")
    create_dual_heatmap(image, outputs, "Original Image Attention", 
                    hidden_image, outputs_hidden, "Hidden Patches Image Attention", 
                    "combined_attention.png")
    
    plot_all_encoder_attentions(image, outputs, "Original Image", "original_attention_all.png")
    plot_all_encoder_attentions(hidden_image, outputs_hidden, "Hidden Patches Image", "hidden_attention_all.png")

    # To plot both encoder and cross attention maps
    # plot_attention_maps(image, outputs, num_layers=3, title_prefix="Layer", base_file_path="original_attention_both")
    # plot_attention_maps(hidden_image, outputs_hidden, num_layers=3, title_prefix="Layer", base_file_path="hidden_attention_both")

if __name__ == "__main__":
    main()