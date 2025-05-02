import argparse
import os

import torch
# Pastikan accelerate diimpor di awal jika Anda ingin menggunakannya secara eksplisit
# (meskipun Trainer sering menanganinya secara implisit saat dijalankan dengan accelerate launch)
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Menonaktifkan WandB (Weights & Biases) jika tidak digunakan
os.environ["WANDB_DISABLED"] = "true"

MODEL_ID = "google/flan-t5-small"


def train(opt):
    # 1. Inisialisasi Accelerator
    #    Meskipun Trainer akan mendeteksinya, inisialisasi eksplisit
    #    berguna untuk kontrol seperti .is_main_process
    accelerator = Accelerator()

    # 2. Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, legacy=False)

    # Konfigurasi Kuantisasi 8-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        # PERBAIKAN: Hapus 'device_map="auto"' untuk DDP standar dengan accelerate.
        #           Accelerate/Trainer akan menangani penempatan device.
        #           device_map="auto" sering bertentangan dengan DDP.
        # device_map="auto",
        quantization_config=quantization_config,
        # use_cache=False diperlukan untuk training
        use_cache=False,
        # torch_dtype bisa diatur, tapi fp16 di TrainingArguments lebih umum
        # torch_dtype=torch.float16,
    )

    # 3. Persiapan Model untuk K-bit Training (PENTING untuk 8-bit)
    #    PERBAIKAN: Baris ini PENTING saat menggunakan kuantisasi (8-bit/4-bit)
    #               dan LoRA/PEFT, terutama jika gradient checkpointing digunakan.
    #               Jangan dikomentari kecuali ada alasan spesifik dan error.
    #               Accelerate seharusnya kompatibel dengan ini.
    model = prepare_model_for_kbit_training(model)

    # 4. Dataset Processing (Sudah terlihat bagus)
    def preprocess_function(examples):
        inputs = examples["comment"]
        targets = examples["label"]

        model_inputs = tokenizer(
            inputs,
            max_length=512,  # Pertimbangkan apakah 512 terlalu panjang/pendek
            padding="max_length",
            truncation=True,
            # Jangan return_tensors="pt" di sini, biarkan Trainer/DataCollator menanganinya
        )

        # Tokenisasi label secara terpisah
        labels = tokenizer(
            targets,
            max_length=5,  # Pastikan panjang max cukup untuk label Anda
            padding="max_length",
            truncation=True,
        )

        # Set ID padding di label menjadi -100
        label_ids = labels["input_ids"]
        # Pastikan konversi ke list atau tensor numpy sebelum perbandingan jika perlu
        processed_labels = []
        for label_row in label_ids:
            processed_row = [label if label != tokenizer.pad_token_id else -100 for label in label_row]
            processed_labels.append(processed_row)

        model_inputs["labels"] = processed_labels
        return model_inputs

    dataset = load_dataset("csv", data_files=opt.dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.20, shuffle=True, seed=42)

    # Menggunakan `with accelerator.main_process_first():` untuk mapping bisa mencegah masalah unduhan/cache ganda
    with accelerator.main_process_first():
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

    # 5. LoRA Configuration (Sudah terlihat bagus)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.3,  # Perhatikan dropout bisa cukup tinggi
        target_modules=["q", "v"],  # Pastikan ini benar untuk Flan-T5
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    # Terapkan LoRA ke model yang sudah disiapkan k-bit
    model = get_peft_model(model, lora_config)

    # PERBAIKAN: Hapus pemindahan manual ke 'cuda'.
    #           Accelerate/Trainer akan menempatkannya ke device yang sesuai per proses.
    # model.to(torch.device("cuda"))

    # Cetak parameter yang dapat dilatih (bagus untuk verifikasi)
    # Gunakan accelerator.print agar hanya dicetak sekali di multi-GPU
    accelerator.print("Trainable parameters:")
    model.print_trainable_parameters()

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=opt.output,  # Ganti ke direktori output yang diinginkan
        eval_strategy="epoch",  # Atau "steps"
        save_strategy="epoch",  # Atau "steps"
        learning_rate=opt.learning_rate,
        gradient_accumulation_steps=2,  # Sesuaikan berdasarkan memori GPU
        auto_find_batch_size=False,  # Set manual lebih baik
        num_train_epochs=opt.epochs,
        save_total_limit=3,  # Batasi jumlah checkpoint yang disimpan
        # PERBAIKAN: 'fp16=True' akan otomatis diaktifkan jika accelerate dikonfigurasi
        #            untuk fp16. Mengaturnya di sini BISA menyebabkan konflik jika
        #            accelerate dikonfigurasi berbeda (misal: no fp16 atau bf16).
        #            Lebih aman MENGHAPUS argumen fp16/bf16 dari TrainingArguments
        #            dan MENGANDALKAN konfigurasi accelerate (`accelerate config`).
        # fp16=opt.fp16,
        report_to="none",  # Atau "tensorboard", "wandb"
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Tentukan metrik untuk model terbaik
        greater_is_better=False,  # Karena kita menggunakan loss
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=opt.batch_size * 2,
        # PERBAIKAN: Hapus argumen tidak valid 'accelerator_config'
        # logging_dir='./logs', # Opsional: Tentukan direktori log TensorBoard
        logging_steps=50,  # Catat metrik setiap N step
        save_steps=500,  # Simpan checkpoint setiap N step (jika save_strategy="steps")
        eval_steps=500,  # Evaluasi setiap N step (jika eval_strategy="steps")
        gradient_checkpointing=True,  # Aktifkan jika memori terbatas, perhatikan implikasi performa
        # Pastikan prepare_model_for_kbit_training dipanggil
        gradient_checkpointing_kwargs={"use_reentrant": False}  # Coba False jika didukung, bisa lebih cepat
    )

    # PERBAIKAN: Hapus pengecekan dan penetapan accelerator_config yang tidak valid
    # if opt.accelerator_config:
    #     training_args.accelerator_config = opt.accelerator_config

    # 7. Trainer Initialization (Sudah terlihat bagus)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Tambahkan tokenizer ke Trainer, berguna untuk beberapa hal
        # data_collator akan dibuat otomatis untuk Seq2Seq jika tokenizer ada
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Berhenti jika eval_loss tidak membaik selama 3 epoch
                early_stopping_threshold=0.0,
            ),
        ],
    )

    # 8. Mulai Training
    accelerator.print("Starting training...")
    trainer.train()
    accelerator.print("Training finished.")

    # 9. Simpan Model Final (Hanya adapter LoRA)
    #    Trainer akan menyimpan checkpoint terbaik jika load_best_model_at_end=True.
    #    Kode ini menyimpan adapter *terakhir* secara eksplisit.
    #    Gunakan wait_for_everyone untuk memastikan semua proses selesai sebelum menyimpan.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Dapatkan model dasar (jika diperlukan, tapi save_pretrained dari PEFT model sudah cukup)
        # unwrapped_model = accelerator.unwrap_model(model)
        final_output_dir = os.path.join(opt.output, "final_adapter")
        model.save_pretrained(final_output_dir)  # Simpan adapter LoRA
        tokenizer.save_pretrained(final_output_dir)  # Simpan juga tokenizer
        accelerator.print(f"Final LoRA adapter saved in {final_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the CSV dataset file.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU device for training.")  # Kurangi default jika memori terbatas
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate.")  # Mungkin perlu lebih kecil untuk fine-tuning
    # parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (configure via accelerate).") # Dihapus, konfig via accelerate
    # parser.add_argument("--accelerator_config", type=str, default=None) # Dihapus
    parser.add_argument("--output", type=str, default="flan-t5-small-lora-finetuned-8bit",
                        help="Output directory for checkpoints and final adapter.")
    args = parser.parse_args()

    # Validasi sederhana
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    print(f"Starting training with arguments: {args}")
    train(args)