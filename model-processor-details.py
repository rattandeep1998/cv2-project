from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
from torchviz import make_dot

model_name = "ahmed-masry/unichart-base-960"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model: {model}")

print(f"Processor: {processor}")

image_path = './data/chartqa/train/png/12044.png'

image = Image.open(image_path).convert("RGB")
prompt = "<extract_data_table> <s_answer>"

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

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.split("<s_answer>")[1].strip()

# generate a model architecture visualization
make_dot(outputs,
         params=dict(model.named_parameters()),
         show_attrs=True,
         show_saved=True).render("MyPyTorchModel", format="png")