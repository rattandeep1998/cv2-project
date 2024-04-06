import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor
from datasets import load_dataset, load_from_disk

added_tokens = []

class VistextDataset(Dataset):
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
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id

        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.images_folder = images_folder

  
        self.dataset = self.load_dataset(json_path)
        self.dataset_length = len(self.dataset)

        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix

    def load_dataset(self, json_path: str):
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        image_name = sample['img_id'] + ".png"

        # input_tensor
        img_path = os.path.join(self.images_folder, image_name)
        img = Image.open(img_path)
        pixel_values = self.processor(img.convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        data_table = sample['table']
        # input_prompt = "<extract_data_table> <s_answer>"
        processed_parse = self.task_prefix + " " + self.prompt_end_token + " " + data_table + self.processor.tokenizer.eos_token 

        print(processed_parse)
        print("===============")
        
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