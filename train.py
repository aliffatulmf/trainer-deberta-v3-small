import argparse
import os
import sys

try:
    import torch
    from accelerate import Accelerator
    from datasets import load_dataset
    from peft import (
        prepare_model_for_kbit_training,
        get_peft_model,
        LoraConfig,
        TaskType,
    )
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install the required packages")
    print(e)
    sys.exit(1)
except Exception as e:
    print("Unexpected error:", e)
    sys.exit(1)

os.environ["WANDB_DISABLED"] = "true"

MODEL_ID = "microsoft/deberta-v3-small"

def train(config):
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quantization_config = BitsAndBytesConfig(load_in_8bit=config["load_in_8bit"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        use_cache=False,
    )

    model = prepare_model_for_kbit_training(model)

    def preprocess_function(examples):
        inputs = examples["comment"]

        model_inputs = tokenizer(
            inputs,
            max_length=config["input_max_length"],
            padding=config["padding"],
            truncation=config["input_truncation"],
            return_tensors=config["return_tensors"],
        )

        # labels = tokenizer(
        #     targets,
        #     max_length=config["target_max_length"],
        #     padding=config["padding"],
        #     truncation=config["target_truncation"],
        #     return_tensors=config["return_tensors"],
        # )
        # labels = labels["input_ids"]
        # labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = torch.tensor(examples["label"])

        return model_inputs

    dataset = load_dataset(config["dataset"])
    # dataset = dataset.map(
    #     preprocess_function,
    #     batched=True,
    #     remove_columns=dataset["train"].column_names,
    # )
    # dataset = dataset["train"].train_test_split(
    #     test_size=config["test_size"],
    #     shuffle=config["shuffle"],
    #     seed=config["seed"]
    # )

    # dataset = dataset["train"].train_test_split(
    #     test_size=config["test_size"], shuffle=config["shuffle"], seed=config["seed"]
    # )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on train dataset",
        )

        eval_dataset = dataset["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["validation"].column_names,
            desc="Running tokenizer on eval dataset",
        )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias=config["lora_bias"],
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)

    accelerator.print("Trainable parameters:")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config["output"],
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        learning_rate=config["learning_rate"],
        auto_find_batch_size=config["auto_find_batch_size"],
        num_train_epochs=config["epochs"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=config["gradient_checkpointing_kwargs"],
        log_level=config["log_level"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"],
                early_stopping_threshold=config["early_stopping_threshold"],
            )
        ],
    )

    accelerator.print("Starting training...")
    trainer.train()
    accelerator.print("Training finished.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_output_dir = os.path.join(config["output"], config["final_adapter_dir"])
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        accelerator.print(f"Final LoRA adapter saved in {final_output_dir}")


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the CSV dataset file."
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU device for training.",
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for checkpoints and final adapter.",
    )
    parser.add_argument("--cpu", action="store_true", help="use CPU instead of GPU.")
    return parser.parse_args()


if __name__ == "__main__":
    opt = opts()

    config = {
        # Command line arguments
        "dataset": opt.dataset,
        "epochs": opt.epochs,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        "output": opt.output,
        "use_cpu": opt.cpu,
        # Model configuration
        # "model_id": "google/flan-t5-small",
        "model_id": MODEL_ID,
        # Accelerator configuration
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 2,
        # Quantization configuration
        "load_in_8bit": True,
        # Tokenizer configuration
        "input_max_length": 256,
        "target_max_length": 3,
        "padding": "max_length",
        "input_truncation": True,
        "target_truncation": False,
        "return_tensors": "pt",
        # Dataset split configuration
        "test_size": 0.20,
        "shuffle": True,
        "seed": 42,
        # LoRA configuration
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.2,
        "lora_target_modules": ["query_proj", "value_proj"],
        "lora_bias": "none",
        # Training arguments
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "auto_find_batch_size": False,
        "save_total_limit": 10,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_batch_size": opt.batch_size * 2,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "log_level": "error",
        # Early stopping configuration
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.0,
        # Output configuration
        "final_adapter_dir": "final_adapter",
    }

    train(config)
