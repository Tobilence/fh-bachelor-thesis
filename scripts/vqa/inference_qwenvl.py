import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Option 1: Distribute the model across multiple GPUs
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto"  # This will automatically distribute across available GPUs
)

# Option 2: Load quantized model (4-bit quantization)
# from transformers import BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     quantization_config=quantization_config,
#     device_map="auto"  # Will distribute across GPUs
# )

# Enable flash attention for better memory efficiency
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    min_pixels=256*28*28,  # 200,704 pixels
    max_pixels=2000*28*28,  # ~1,568,000 pixels - covers most of your image size
    devive_map="auto"
)
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///home/student/fh-bachelor-thesis/data/wood-defects-parsed/images/val/103500057.jpg",
            },
            {"type": "text", "text": "What do you see in this image?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to the device where the first layer of the model is
device = model.device if hasattr(model, 'device') else next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Inference with lower memory usage
with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Use mixed precision
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)