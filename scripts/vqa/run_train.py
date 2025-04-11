import torch
import wandb
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
import json
import argparse
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from datetime import datetime
import os
from pathlib import Path
import logging

def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def setup_logging(run_name):
    logs_dir = get_project_root() / "logs" / "qwen" / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(logs_dir / "train.log"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, 
                       default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return parser.parse_args()

def load_dataset(split: str):
    project_root = get_project_root()
    dataset_path = project_root / f"data/wood-defects-parsed/qwen/{split}.json"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    # Extract only the messages from each item in the dataset
    messages_only = [item["messages"] for item in dataset]
        
    return messages_only

def setup_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model, processor, peft_config

def setup_training():

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="qwen-finetune-v0",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=16,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    return training_args



def main(args):
    setup_logging(args.run_name)
    logging.info(f"Starting training run: {args.run_name}")

    model, processor, peft_config = setup_model("Qwen/Qwen2.5-VL-7B-Instruct")
    training_args = setup_training()

    wandb.init(
        project="qwen_finetune",  # change this
        name=args.run_name,  # change this
        config=training_args,
    )

    def collate_fn(batch):
        # 🔧 FIX: batch is a list of lists of messages. We flatten each one.
        # Now `examples` becomes a list of dicts, each with a `.messages` field.
        examples = [{"messages": example} for example in batch]

        input_messages = [ex["messages"][:-1] for ex in examples]  # Keep system + user
        target_messages = [ex["messages"][-1] for ex in examples]  # Just the assistant

        # ✅ Prompt-only for input
        prompts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in input_messages
        ]

        # ✅ Full convo (input + expected response) for label alignment
        full_conversations = [
            processor.apply_chat_template(ex["messages"], tokenize=False)
            for ex in examples
        ]

        # ✅ Extract image tensors
        image_inputs = [process_vision_info(ex["messages"])[0] for ex in examples]

        # ✅ Tokenize with processor
        batch = processor(
            text=full_conversations,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()

        # ✅ Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # ✅ Mask image tokens
        if isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        # ✅ Mask prompt (non-target) tokens
        for i, (prompt_text, full_text) in enumerate(zip(prompts, full_conversations)):
            prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch



    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=load_dataset("train"),
        eval_dataset=load_dataset("val"),
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    print("clearing cuda cache...")
    torch.cuda.empty_cache()
    print("Cleared! ")
    args = parse_arguments()
    main(args)
