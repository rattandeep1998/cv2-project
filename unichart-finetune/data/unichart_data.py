import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor
from datasets import load_dataset, load_from_disk
import numpy as np

added_tokens = []

def hide_patches(image, grid_size=(16, 16), p_hide=0.3, average_pixel_value=128):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    patch_h, patch_w = h // grid_size[0], w // grid_size[1]

    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            if np.random.rand() < p_hide:  # Randomly choose to hide this patch based on p_hide
                img_array[i:i + patch_h, j:j + patch_w, :] = average_pixel_value
    
    return Image.fromarray(img_array)

class UnichartDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        images_folder: str,
        max_length: int,
        processor : DonutProcessor = None,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        task_prefix: str = '<chartqa>',
        sort_json_key: bool = True,
        use_hide_patches: bool = False,
        p_hide: float = 0.3,
        grid_size: Tuple[int, int] = (16, 16),
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.use_hide_patches = use_hide_patches
        self.p_hide = p_hide
        self.grid_size = grid_size

        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.images_folder = images_folder

  
        self.dataset = self.load_dataset(json_path)
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix

        self.average_pixel_value = self.calculate_average_pixel_value() if self.use_hide_patches else None
        print(f"Average pixel value: {self.average_pixel_value}")

    def load_dataset(self, json_path: str):
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def calculate_average_pixel_value(self):
        # Calculate the average pixel value across a subset of the dataset
        num_samples = min(10, len(self.dataset))  # use 10 images to estimate the average pixel value
        total = 0
        for i in range(num_samples):
            image_path = os.path.join(self.images_folder, self.dataset[i]['img_id'] + ".png")
            image = Image.open(image_path).convert("RGB")
            total += np.mean(np.array(image))
        return total / num_samples
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        image_name = sample['img_id'] + ".png"

        # input_tensor
        img_path = os.path.join(self.images_folder, image_name)
        img = Image.open(img_path).convert("RGB")

        if self.split == "train" and self.use_hide_patches:
            img = hide_patches(img, self.grid_size, self.p_hide, self.average_pixel_value)

        pixel_values = self.processor(img, random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        data_table = sample['table']
        # input_prompt = "<extract_data_table> <s_answer>"
        processed_parse = self.task_prefix + " " + self.prompt_end_token + " " + data_table + self.processor.tokenizer.eos_token 
        
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt 
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse