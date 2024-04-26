from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re

model_name = "ahmed-masry/unichart-base-960"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model: {model}")

print(f"Processor: {processor}")