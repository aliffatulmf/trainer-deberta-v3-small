import argparse

import torch
import os

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, Trainer,
                          EarlyStoppingCallback)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

os.environ["WANDB_DISABLED"] = "true"

MODEL_ID = "google/flan-t5-small"


def train(opt):
    # Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        ),
        use_cache=False,
        torch_dtype=torch.float16,
    )

    # model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    model = prepare_model_for_kbit_training(model) # disable for multi-gpu train

    # Dataset
    def preprocess_function(examples):
        inputs = examples["comment"]
        targets = examples["label"]

        model_inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            targets,
            max_length=5,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs

    dataset = load_dataset("csv", data_files=opt.dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.20, shuffle=True, seed=42)

    train_dataset = split_dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = split_dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=split_dataset["test"].column_names,
        desc="Running tokenizer on eval dataset",
    )

    # LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.3,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.to(torch.device("cuda"))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="runs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=opt.learning_rate,
        gradient_accumulation_steps=2,
        auto_find_batch_size=False,
        num_train_epochs=opt.epochs,
        save_total_limit=8,
        fp16=opt.fp16,
        report_to="none",
        load_best_model_at_end=True,
        greater_is_better=False,
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=opt.batch_size * 2,
    )

    if opt.accelerator_config:
        training_args.accelerator_config = opt.accelerator_config

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0,
            ),
        ],
        n_gpu=opt.n_gpu,
    )

    trainer.train()

    model.save_pretrained(opt.output)
    tokenizer.save_pretrained(opt.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--accelerator_config", type=str, default=None)
    parser.add_argument("--output", type=str, default="flan-t5-small-lora-finetuned-gambling-8bit")
    parser.add_argument("--n_gpu", type=int, default=1)
    args = parser.parse_args()

    train(args)
