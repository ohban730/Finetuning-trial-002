# Finetuning-trial-002

指示ファインチューニングにおける「フルファインチューニング vs LoRA」の計算リソース比較検証。
題材はカジュアルな日本語文 → フォーマルな日本語文への変換（片方向）。

## 関連リンク

- 検証結果まとめ: [Zenn記事](https://zenn.dev/sapolas/articles/0175b65f1e5b8c)
- 学習済みモデル（Hugging Face）:
  - Full-FT: [Sapolas0730/japanese-gpt2-medium-formal-fullft](https://huggingface.co/Sapolas0730/japanese-gpt2-medium-formal-fullft)
  - LoRA: [Sapolas0730/japanese-gpt2-medium-formal-lora](https://huggingface.co/Sapolas0730/japanese-gpt2-medium-formal-lora)

## 環境

- conda環境: `llm-sandbox`
- Python 3.10 / PyTorch (CUDA) / transformers / datasets / accelerate / matplotlib / sentencepiece

```bash
conda activate llm-sandbox
```

以降のコマンドは全てこの環境のPythonで実行する（`python`が別環境を指す場合はフルパスを使う）。

```bash
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" <script>.py ...
```

## パイプライン

```
magpie.py（データ生成） → main.py / main_lora.py（学習） → infer.py（動作確認） → report.py（比較レポート）
```

用語（エポック、学習率、オプティマイザー、PEFT/LoRA、Early Stoppingなど）でつまずいたら [docs/glossary.md](docs/glossary.md) を参照。

### 1. データ生成: `magpie.py`

`Qwen/Qwen2.5-7B-Instruct` を使い、MAGPIE手法でカジュアル文→フォーマル文変換の指示データを生成する。system promptはペルソナ説明のみに留め、userターンをモデルに自己生成させることで多様な指示文を作る。日本語純度・フォーマット・長さ比・疑問形保持・重複のフィルタを通過したものだけを採用する。

```bash
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" magpie.py --num_samples 1000 --output_path dataset_formal.json
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--num_samples` | `50` | 生成する件数 |
| `--output_path` | `dataset_formal.json` | 出力先JSON（`instruction`/`input`/`output`形式）。10件ごとに途中経過を保存するので中断しても失われない |

`Qwen/Qwen2.5-7B-Instruct`はゲートなしで利用できるため、`huggingface-cli login`やライセンス同意は不要（元は`meta-llama/Meta-Llama-3-8B-Instruct`を使用していたが、そのライセンス条項の懸念から変更した。詳細は後述の「ライセンス」参照）。

> **注意**: モデルによって自己生成する「userターン」の形式が異なる。Llama-3系は`「次の文をフォーマルな言い方に変換してください：（文）」`という依頼文形式で生成したが、Qwen2.5はこの依頼文を使わずカジュアルな一文を直接生成する傾向がある。`extract_casual_sentence`はどちらの形式にも対応している。

### 2. 学習: `main.py`

`rinna/japanese-gpt2-medium`（GPT-2アーキテクチャ・336M・日本語ネイティブトークナイザー）をフルファインチューニングする。検証データを分けて毎エポックeval_lossを記録し、Early Stoppingで最良時点のモデルを自動採用する。中間チェックポイントは学習後に削除し、最終モデル1つだけをディスクに残す。

```bash
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" main.py --data_path dataset_formal.json --output_dir output/full_ft --epochs 15
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--data_path` | `dataset_formal.json` | 学習データ（`instruction`/`input`/`output`形式） |
| `--output_dir` | `output/full_ft` | モデル・トークナイザー・`metrics.json`の保存先 |
| `--epochs` | `3.0` | 最大エポック数（Early Stoppingで途中打ち切りされ得る） |
| `--batch_size` | `4` | `per_device_train_batch_size` |
| `--grad_accum` | `4` | `gradient_accumulation_steps` |
| `--lr` | `5e-5` | 学習率 |
| `--max_length` | `512` | 最大トークン長 |
| `--val_ratio` | `0.1` | 検証用に分けるデータの割合。`0`でholdoutなし |
| `--early_stopping_patience` | `3` | 検証損失がN回連続で改善しなければ打ち切り。`0`で無効化（最大エポックまで回す） |

学習後、`output_dir/metrics.json`に以下が保存される。

- `trainable_params` / `total_params`: 学習可能パラメータ数（LoRA版との比較に使う）
- `peak_gpu_memory_gb`: ピークGPUメモリ使用量
- `train_runtime_sec`: 学習時間
- `final_loss`: 最終学習損失
- `best_checkpoint`: Early Stoppingで採用したモデルの`step` / `epoch` / `eval_loss`（該当なしなら`null`）
- `log_history`: 損失曲線の生データ（`report.py`が読む）

### 2b. 学習（LoRA版）: `main_lora.py`

`main.py`と同じデータパイプライン・プロンプトテンプレート・検証分割・Early Stoppingの仕組みを使い、`peft`でLoRAアダプタのみを学習する（ベースモデルは凍結）。`--data_path`, `--output_dir`, `--epochs`, `--batch_size`, `--grad_accum`, `--lr`, `--max_length`, `--val_ratio`, `--early_stopping_patience`は`main.py`と共通。

```bash
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" main_lora.py --data_path dataset_formal.json --output_dir output/lora --epochs 15
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--lr` | `1e-4` | LoRAはFull-FTより高めの学習率が定石なため、`main.py`（`5e-5`）とデフォルト値を変えている |
| `--lora_r` | `8` | LoRAの低ランク行列の次元数 |
| `--lora_alpha` | `16` | LoRAのスケーリング係数 |
| `--lora_dropout` | `0.05` | LoRA層のdropout |
| `--lora_target_modules` | `c_attn,c_proj` | LoRAを適用する層（GPT-2はConv1D層） |

`metrics.json`は`main.py`と同じスキーマ＋`lora_config`が追加された形で保存されるので、そのまま`report.py`で比較できる。保存されるのはLoRAアダプタのみ（`adapter_model.safetensors`、数MB程度）で、ベースモデルは保存しないためFull-FTよりディスク使用量が大幅に少ない。

実データ1000件・最大15epochでの実行結果: 学習可能パラメータ 2,162,688（Full-FTの0.64%）、ピークGPUメモリ 3.17GB（Full-FTは6.84GB）、アダプタサイズ約8.7MB。詳細は[Zenn記事](https://zenn.dev/sapolas/articles/0175b65f1e5b8c)を参照。

### 3. 動作確認: `infer.py`

`main.py`（Full-FT）・`main_lora.py`（LoRA）どちらの成果物からもカジュアル文をフォーマル文に変換できる。学習時と同じ`build_prompt`を`main.py`から直接importして使うため、プロンプト形式がズレない。`--model_dir`に`adapter_config.json`があるかどうかで自動判定し、LoRAの場合は`AutoPeftModelForCausalLM`でベースモデルを自動解決してアダプタを適用する。

```bash
# Full-FTモデルで変換
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" infer.py --model_dir output/full_ft --text "明日暇？"

# LoRAアダプタで変換
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" infer.py --model_dir output/lora --text "明日暇？"

# 対話モード（--textを省略）
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" infer.py --model_dir output/full_ft
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--model_dir` | `output/full_ft` | `main.py`または`main_lora.py`の`--output_dir`で保存したモデルのパス |
| `--text` | なし | 変換したいカジュアルな文。省略時は対話モード（`exit`かCtrl+Cで終了） |
| `--max_new_tokens` | `100` | 生成する最大トークン数 |
| `--sample` | オフ | 指定するとサンプリング生成（temperature=0.7, top_p=0.9）。デフォルトはgreedy |

> **注意**: `min_new_tokens=5`を指定し、少なくとも5トークンは生成してからEOSでの停止を許可している。学習が浅いモデルはgreedy decodingで即座にEOSを出して空文字列になることがあるため（実際に発生を確認済み）。

### 4. 比較レポート: `report.py`

複数の学習run（Full-FTとLoRAなど）の`metrics.json`を読み、比較表と損失曲線グラフを生成する。損失曲線にはEarly Stoppingで採用したモデルの位置を★で表示する。

```bash
& "C:\Users\owner\miniconda3\envs\llm-sandbox\python.exe" report.py --run "Full-FT:output/full_ft/metrics.json" --run "LoRA:output/lora/metrics.json" --output_dir report
```

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--run` | 必須（複数指定可） | `表示名:metrics.jsonのパス` の形式。例: `Full-FT:output/full_ft/metrics.json` |
| `--output_dir` | `report` | 出力先ディレクトリ |

出力:

- `comparison_table.csv` / `comparison_table.md`: 学習可能パラメータ数、GPUメモリ、学習時間、最終損失、採用エポック/検証損失の比較表（Zenn記事にそのまま貼れる）
- `loss_curve.png`: train/eval損失曲線（複数run指定で重ねて比較可能）

## ファイル構成

```
magpie.py     データ生成（MAGPIE）
main.py       フルファインチューニング（プロンプトテンプレートはinfer.py/main_lora.pyと共有）
main_lora.py  LoRAファインチューニング（main.pyのデータパイプラインを再利用）
infer.py      学習済みモデル（Full-FT/LoRAどちらも対応）での推論確認
report.py     複数runの比較表・グラフ生成
dataset_formal.json  生成済みデータセット
docs/glossary.md     ファインチューニング用語集
output/       学習成果物（モデル・metrics.json）。git管理対象外を推奨
report/       比較表・グラフの出力先
```

## 今後改善するなら

Full-FTではエポック数が大きくなると過学習が起きた（エポック2をピークにeval_lossが悪化）。学習データが1000件と少なかったことが一因と考えられるため、より大量の学習データを用意して再検証したい。詳細な考察は[Zenn記事](https://zenn.dev/sapolas/articles/0175b65f1e5b8c)を参照。

## ライセンス

### 本プロジェクトのコード

MIT License。詳細は[LICENSE](LICENSE)を参照。

### ベースモデル

`rinna/japanese-gpt2-medium`はMIT Licenseで配布されています。本プロジェクトが公開しているファインチューニング済みモデル（Full-FT / LoRAアダプタ）もこれに従います。

### 学習データ（`dataset_formal.json`）について

学習データは`Qwen/Qwen2.5-7B-Instruct`（Apache License 2.0、ゲートなし）の出力をMAGPIE手法で合成したものです。

このプロジェクトは当初、データ生成に`meta-llama/Meta-Llama-3-8B-Instruct`を使用していました。しかし同モデルが従うMeta Llama 3 Community Licenseの第1.b.v項に、

> "You will not use the Llama Materials or any output or results of the Llama Materials to improve any other large language model (excluding Meta Llama 3 or derivative works thereof)."

という「Llama 3の出力を使って他のLLMを改善（ファインチューニング）してはならない」という制限があることに後から気づき、本プロジェクトの用途がこの制限に文面上抵触する可能性があったため、より寛容なライセンス（Apache 2.0）で提供されている`Qwen/Qwen2.5-7B-Instruct`にデータ生成モデルを切り替え、データ・学習済みモデルを作り直しました。他のモデルの出力を学習データに使う際は、ベースモデルのライセンス条項（特に生成物の再利用・再配布・派生モデルの学習に関する制限）を事前に確認することをお勧めします。
