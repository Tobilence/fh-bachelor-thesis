import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel, PeftConfig
from qwen_vl_utils import process_vision_info

################################################################################
### Model Setup Complete
################################################################################

# Load the base model first
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define adapter path
adapter_path = "/home/student/fh-bachelor-thesis/qwen-finetune-v0/checkpoint-160"

# Load the PEFT config
peft_config = PeftConfig.from_pretrained(adapter_path)

# Load the fine-tuned model
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=256*28*28,
    max_pixels=2000*28*28,
    device_map="auto"
)


################################################################################
### Inference
################################################################################

from pathlib import Path
import json
# Load test data
TEST_SET_PATH = "./data/wood-defects-parsed/qwen/test.json"
file = Path(TEST_SET_PATH)

with open(file, "r") as f:
    test_set = json.load(f)

# Store results
results = []
print(f"Processing {len(test_set)} samples...")

# Process just one sample
samples = test_set[:10]  # Take first sample
for sample in samples:
    input_messages = sample["messages"][:2]
    ground_truth = sample["messages"][2]

    # Prepare inputs
    text = processor.apply_chat_template(
        input_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(input_messages)
    print("Prompt Text: ", text)

    print("Image Inputs: ", image_inputs)
    assert image_inputs is not None and len(image_inputs) > 0, "Image input is missing or invalid"
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to model device
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate with mixed precision
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Process output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]

    print("Generated IDs (trimmed): ", generated_ids_trimmed)
    for ids in generated_ids_trimmed:
        print(ids)
    print("--------------------------------")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"\nProcessed image: {sample['id']}")
    print("Model output:", output_text[0])
    # print("Ground truth:", ground_truth["content"])

    results.append({
        "img_id": sample["id"],
        "output": output_text[0],
        "ground_truth": ground_truth["content"]
    })

# Save results
output_file = "finetuned_inference_results.json"
json.dump(results, open(output_file, "w+"))
print(f"\nResults saved to {output_file}")
