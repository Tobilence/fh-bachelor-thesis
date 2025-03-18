# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())


"""
Grok FInetune Code: 

import requests
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# Move model to GPU
model.to("cuda")

# Load and preprocess dataset
def preprocess_function(examples):
    images = [Image.open(img_path).convert("RGB") for img_path in examples["image"]]
    questions = examples["question"]
    answers = examples["answer"]
    
    # Process inputs
    inputs = processor(images=images, text=questions, padding="max_length", return_tensors="pt")
    # Process labels (answers)
    labels = processor(text=answers, padding="max_length", return_tensors="pt").input_ids
    
    # Replace padding token id with -100 to ignore in loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    inputs["labels"] = labels
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./blip2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    gradient_accumulation_steps=4,  # If batch size is too small for your GPU
    learning_rate=5e-5,
    save_steps=500,
    logging_steps=100,
    fp16=True,  # Mixed precision for faster training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./blip2-finetuned")
processor.save_pretrained("./blip2-finetuned")
"""


"""
GROK Test Finetune Code:

# Load fine-tuned model
processor = Blip2Processor.from_pretrained("./blip2-finetuned")
model = Blip2ForConditionalGeneration.from_pretrained("./blip2-finetuned", device_map="auto").to("cuda")

# Test with an image and question
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
question = "how many dogs are in the picture?"

inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
"""