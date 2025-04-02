from pathlib import Path
import json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

TRAIN_SET_PATH = "data/wood-defects-parsed/qwen/train.json"
file = Path(TRAIN_SET_PATH)

with open(file, "r") as f:
    test_set = json.load(f)


for sample in test_set[:5]:
    input_messages = sample["messages"][:2]
    ground_truth = sample["messages"][2]
    print(input_messages)
    print(ground_truth)
    image_inputs, video_inputs = process_vision_info(input_messages)
    print("img")
    print(image_inputs)