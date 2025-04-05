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

TRAIN_SET_PATH = "./data/wood-defects-parsed/qwen/train.json"
file = Path(TRAIN_SET_PATH)

with open(file, "r") as f:
    test_set = json.load(f)


result = []
print(len(test_set))

for sample in test_set[:3]:
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
    print("Finished inference for: ", sample["id"])
    print("Output: ", output_text)
    result.append({
        "img_id": sample["id"],
        "output": output_text,
    })

json.dump(result, open("inference_result.json", "w+"))



"""
Huggingface example for inference function:
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

"""