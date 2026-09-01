import argparse
import os

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import build_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_lora_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if is_lora_adapter:
        # main_lora.pyの成果物はアダプタのみ保存されているため、
        # adapter_config.json内のbase_model_name_or_pathからベースモデルを自動解決して読み込む
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def convert(tokenizer, model, casual_text: str, max_new_tokens: int, do_sample: bool) -> str:
    instruction = f"次の文をフォーマルな言い方に変換してください：{casual_text}"
    prompt = build_prompt({"instruction": instruction, "input": ""})

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.9 if do_sample else None,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(5, max_new_tokens),  # greedy decodingが即座にEOSを出して空文字列になるのを防ぐ
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="output/full_ft", help="main.pyの--output_dirで保存したモデルのパス")
    parser.add_argument("--text", help="変換したいカジュアルな文。省略時は対話モードになる")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--sample", action="store_true", help="指定するとサンプリング生成（デフォルトはgreedy）")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)

    if args.text:
        print(convert(tokenizer, model, args.text, args.max_new_tokens, args.sample))
        return

    print("カジュアルな文を入力してください（終了は Ctrl+C か 'exit'）")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == "exit":
            break
        print(convert(tokenizer, model, text, args.max_new_tokens, args.sample))


if __name__ == "__main__":
    main()
