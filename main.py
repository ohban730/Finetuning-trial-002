import argparse
import glob
import json
import os
import shutil
import time
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "rinna/japanese-gpt2-medium"  # GPT-2アーキテクチャ・336M・日本語ネイティブトークナイザー

PROMPT_TEMPLATE = (
    "以下はタスクを説明する指示です。指示に従って応答を書いてください。\n\n"
    "### 指示:\n{instruction}\n\n### 応答:\n"
)
PROMPT_TEMPLATE_WITH_INPUT = (
    "以下はタスクを説明する指示と、それに対する入力です。指示に従って応答を書いてください。\n\n"
    "### 指示:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:\n"
)


def build_prompt(example: dict) -> str:
    if example.get("input"):
        return PROMPT_TEMPLATE_WITH_INPUT.format(instruction=example["instruction"], input=example["input"])
    return PROMPT_TEMPLATE.format(instruction=example["instruction"])


def load_dataset(data_path: str, tokenizer, max_length: int) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in raw:
        prompt = build_prompt(item)
        response = item["output"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[:max_length]
        # プロンプト部分は損失計算から除外し、応答部分のみを学習対象にする
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]

        examples["input_ids"].append(input_ids)
        examples["attention_mask"].append([1] * len(input_ids))
        examples["labels"].append(labels)

    return Dataset.from_dict(examples)


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset_formal.json")
    parser.add_argument("--output_dir", default="output/full_ft")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="検証用に分けるデータの割合。0でholdoutなし")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="検証損失がN回連続で改善しなければ打ち切り、最良時点のモデルを採用する。0で無効化",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

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

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 学習中に一時保持したcheckpoint-*は最終モデルを保存済みなので削除する
    for checkpoint_dir in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    main()
