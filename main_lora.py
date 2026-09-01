import argparse
import glob
import json
import os
import shutil
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from main import MODEL_NAME, PadCollator, count_trainable_params, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset_formal.json")
    parser.add_argument("--output_dir", default="output/lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    # LoRAは動かすパラメータが少ない分、Full-FTより高めの学習率が定石
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="検証用に分けるデータの割合。0でholdoutなし")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="検証損失がN回連続で改善しなければ打ち切り、最良時点のモデルを採用する。0で無効化",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRAの低ランク行列の次元数")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRAのスケーリング係数")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="c_attn,c_proj",
        help="LoRAを適用する層をカンマ区切りで指定（GPT-2はConv1D層なのでc_attn/c_proj）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    dataset = load_dataset(args.data_path, tokenizer, args.max_length)
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    eval_dataset = None
    if args.val_ratio > 0:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset = dataset

    trainable_params = count_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"学習可能パラメータ数: {trainable_params:,} / 総パラメータ数: {total_params:,}")

    use_best_model = eval_dataset is not None and args.early_stopping_patience > 0
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        # 学習中は検証損失最良時点のcheckpointだけ一時保持し、終了後にmodel1個だけ残して削除する
        save_strategy="epoch" if use_best_model else "no",
        save_total_limit=1,
        load_best_model_at_end=use_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    callbacks = []
    if use_best_model:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    best_checkpoint = None
    if use_best_model and trainer.state.best_model_checkpoint is not None:
        best_step = int(trainer.state.best_model_checkpoint.rstrip("/\\").split("-")[-1])
        best_epoch = next(
            (e["epoch"] for e in trainer.state.log_history if e.get("step") == best_step and "eval_loss" in e),
            None,
        )
        best_checkpoint = {
            "step": best_step,
            "epoch": best_epoch,
            "eval_loss": trainer.state.best_metric,
        }

    metrics = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_runtime_sec": elapsed,
        "peak_gpu_memory_gb": peak_memory_gb,
        "final_loss": train_result.training_loss,
        "best_checkpoint": best_checkpoint,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules.split(","),
        },
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"学習時間: {elapsed:.1f}秒")
    if peak_memory_gb is not None:
        print(f"ピークGPUメモリ使用量: {peak_memory_gb:.2f} GB")
    print(f"最終損失: {train_result.training_loss:.4f}")
    if use_best_model and trainer.state.best_metric is not None:
        print(f"採用したモデル: {trainer.state.best_model_checkpoint}（検証損失={trainer.state.best_metric:.4f}）")

    # PeftModelなのでLoRAアダプタのみ保存される（ベースモデルは保存しないためFull-FTよりずっと軽い）
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    for checkpoint_dir in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    main()
