import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# カジュアルな一文の題材にバリエーションを持たせるためのトピック一覧
TOPICS = [
    "友人との遊びの誘い",
    "仕事の進捗報告",
    "家族への連絡",
    "体調に関する近況報告",
    "待ち合わせ場所の相談",
    "SNSでの近況シェア",
    "アルバイトのシフト相談",
    "趣味の話題",
    "旅行の計画",
    "食事の誘い",
]

# システムプロンプトはアシスタントの役割・ドメインの説明のみに留め、
# 具体的な変換対象の文は書かない（ここに実例を書くとuserターン生成時に
# モデルが「回答者」の口調で応答してしまい、役割が崩れる）
SYSTEM_PROMPT_TEMPLATE = (
    "あなたは日本語の文章をカジュアルな言い回しからフォーマルな言い回しに変換するアシスタントです。"
    "ユーザーは{topic}に関するカジュアルな日本語の一文を挙げたうえで、"
    "「次の文をフォーマルな言い方に変換してください：（カジュアルな文）」という形式で依頼してきます。"
    "あなたは依頼された文だけを、丁寧で自然な日本語の一文に変換して回答します。"
    "説明や前置き、英語は一切使わず、変換後の日本語の文のみを答えます。"
)
PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
)
MAX_NEW_TOKENS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

JP_CHAR_PATTERN = re.compile(r"[぀-ヿ一-鿿]")
ASCII_LETTER_PATTERN = re.compile(r"[A-Za-z]")
INSTRUCTION_PATTERN = re.compile(r"次の文をフォーマルな言い方に変換してください[：:]\s*(.+)", re.DOTALL)
QUOTE_CHARS = "「」『』\"'“”‘’ 　"


def is_japanese_text(text: str, min_jp_ratio: float = 0.3, max_ascii_ratio: float = 0.2) -> bool:
    text = text.strip()
    if not text:
        return False
    jp_count = len(JP_CHAR_PATTERN.findall(text))
    ascii_count = len(ASCII_LETTER_PATTERN.findall(text))
    total = len(text)
    return (jp_count / total) >= min_jp_ratio and (ascii_count / total) <= max_ascii_ratio


def extract_casual_sentence(instruction: str) -> str | None:
    """instructionが規定のフォーマットに沿っているか確認し、カジュアルな原文部分だけ取り出す。
    フォーマットから外れた（=文体変換タスクではない）生成結果を除外するためのフィルタ。"""
    match = INSTRUCTION_PATTERN.match(instruction.strip())
    if not match:
        return None
    return match.group(1).strip(QUOTE_CHARS)


def filter_and_normalize(instruction: str, output: str) -> tuple[str, str] | None:
    casual = extract_casual_sentence(instruction)
    if not casual:
        return None
    output = output.strip(QUOTE_CHARS)
    if not output:
        return None

    # 極端に短い/長い変換（要約しすぎ・冗長すぎ）を除外
    length_ratio = len(output) / len(casual)
    if not (0.4 <= length_ratio <= 3.0):
        return None

    # 原文が疑問文なら、変換後も疑問の体裁が保たれているか確認
    if casual.endswith(("？", "?")) and not any(c in output[-4:] for c in ("か", "？", "?")):
        return None

    normalized_instruction = f"次の文をフォーマルな言い方に変換してください：{casual}"
    return normalized_instruction, output


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, **generate_kwargs) -> tuple[str, str]:
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text_gen_only = tokenizer.decode(
        output_ids[0][input_ids.shape[-1] :], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return full_text, text_gen_only


def generate_one(tokenizer, model) -> dict | None:
    eot_str = "<|eot_id|>"
    topic = random.choice(TOPICS)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(topic=topic)
    prompt = PROMPT_TEMPLATE.format(system_prompt=system_prompt)

    sys_usr, instruction = generate(tokenizer, model, prompt)
    instruction = instruction.replace(eot_str, "").strip()
    if not is_japanese_text(instruction):
        return None
    if not sys_usr.endswith(eot_str):
        sys_usr += eot_str

    response_gen_input = sys_usr + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    _, output = generate(tokenizer, model, response_gen_input)
    output = output.replace(eot_str, "").strip()
    if not is_japanese_text(output):
        return None

    normalized = filter_and_normalize(instruction, output)
    if normalized is None:
        return None
    instruction, output = normalized

    return {"instruction": instruction, "input": "", "output": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_path", default="dataset_formal.json")
    args = parser.parse_args()

    tokenizer, model = load_model()
    print("モデルとトークナイザーのロードが完了しました。")

    dataset_list = []
    seen_casuals = set()
    attempts = 0
    duplicates = 0
    max_attempts = args.num_samples * 5
    while len(dataset_list) < args.num_samples and attempts < max_attempts:
        attempts += 1
        example = generate_one(tokenizer, model)
        if example is None:
            continue

        casual = extract_casual_sentence(example["instruction"])
        if casual in seen_casuals:
            duplicates += 1
            continue
        seen_casuals.add(casual)

        dataset_list.append(example)
        if len(dataset_list) % 10 == 0:
            print(f"Generated {len(dataset_list)} samples")
            # 大量生成時に中断しても進捗を失わないよう定期的に保存
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(dataset_list, f, ensure_ascii=False, indent=4)

    print(f"最終的に{len(dataset_list)}件生成しました（試行回数: {attempts}, 重複除外: {duplicates}）。")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
