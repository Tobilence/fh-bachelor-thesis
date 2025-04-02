import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto"  # This will automatically distribute across available GPUs
)

# default processer
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", 
    min_pixels=256*28*28,  # 200,704 pixels
    max_pixels=2000*28*28,  # ~1,568,000 pixels - covers most of your image size
    devive_map="auto"
)


from pathlib import Path
import json

TRAIN_SET_PATH = "/home/student/fh-bachelor-thesis/data/wood-defects-parsed/vqa/train.json"
file = Path(TRAIN_SET_PATH)

with open(file, "r") as f:
    test_set = json.load(f)


result = []
print(len(test_set))

for sample in test_set:
    input_messages = sample["messages"][:2]
    ground_truth = sample["messages"][2]
    # Preparation for inference
    text = processor.apply_chat_template(
        input_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(input_messages)
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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("--------------------------------")
    print("ID: ", sample["id"])
    print("INPUT: ", sample["messages"])
    print("OUTPUT: ", output_text)
    print("GROUND TRUTH: ", ground_truth)
    print("--------------------------------")
    result.append(output_text)

print(result)