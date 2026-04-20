from __future__ import annotations

import argparse
import random

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="josecannete/large_spanish_corpus")
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.20,
        help="Fraction of the training split used for LoRA fine-tuning. The paper setting is 0.20.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./lora-finetuned-Spanish-llama-3-8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def preprocess_function(examples, tokenizer, max_length):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


def main():
    args = parse_args()
    random.seed(args.seed)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)

    print(f"Selecting {args.subset_ratio:.0%} of the dataset...")
    subset_size = int(args.subset_ratio * len(dataset["train"]))
    indices = random.sample(range(len(dataset["train"])), subset_size)
    train_subset = dataset["train"].select(indices)

    print("Splitting dataset into train and validation sets...")
    train_test_split = train_subset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing dataset with multi-processing...")
    train_dataset = train_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    print("Freezing model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.use_wandb:
        wandb.init(
            project="lora-llama3-8b",
            name="finetune-large-spanish-corpus",
            config={
                "model_name": args.model_name,
                "dataset": args.dataset_name,
                "subset_ratio": args.subset_ratio,
            },
        )

    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=500,
        save_steps=10000,
        eval_steps=10000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        report_to="wandb" if args.use_wandb else "none",
        logging_dir="./logs",
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training model...")
    trainer.train()

    print("Saving fine-tuned model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
