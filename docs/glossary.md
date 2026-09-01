# ファインチューニング用語集

`main.py` / `main_lora.py` を読みながら出てくる用語の意味と、このプロジェクトでの実際の値を対応付けてまとめたもの。

---

## 1. 学習の基本用語（おさらい）

### エポック（epoch）
学習データ全体を1周すること。`--epochs 15`なら最大15周する（ただしEarly Stoppingで途中打ち切りされることがある）。

### バッチサイズ / 勾配累積 / 実効バッチサイズ
- **バッチサイズ**（`--batch_size`）: 1回の重み更新に使うサンプル数。一度にGPUメモリに載る量。
- **勾配累積**（`--grad_accum`）: バッチサイズを増やさずに「実質的な」バッチサイズを増やすテクニック。N回分の勾配を溜めてから1回だけ重みを更新する。
- **実効バッチサイズ** = `batch_size × grad_accum`。例えば`--batch_size 4 --grad_accum 4`なら実効バッチサイズは16。GPUメモリが足りなくて`batch_size`を上げられないときに、`grad_accum`を上げることで疑似的に大きいバッチで学習できる。

### 学習率（learning rate, lr）
1回の重み更新でどれだけ重みを動かすかの大きさ。大きすぎると発散（lossがNaNや爆発）、小さすぎるとほとんど学習が進まない。`main.py`のデフォルトは`5e-5`（Full-FT）、`main_lora.py`のデフォルトは`1e-4`（LoRA。理由は後述）。

学習率は学習が進むにつれて減衰していく（HF Trainerのデフォルトはlinear decay）。ログの`learning_rate`が徐々に小さくなっていくのはこのため。

### オプティマイザー（optimizer）
重みをどう更新するかのアルゴリズム。`Trainer`はデフォルトで**AdamW**を使う。

- 単純な勾配降下法（`重み -= 学習率 × 勾配`）だと、勾配のスケールが層やパラメータごとにバラバラで学習が不安定になりやすい
- Adamは各パラメータごとに「勾配の移動平均（向き）」と「勾配の2乗の移動平均（大きさ）」を追跡し、パラメータごとに実質的な学習率を自動調整する
- AdamWはAdamに「重み減衰（weight decay、正則化の一種）」を正しく組み込んだ改良版

覚えなくていいポイント: **AdamWは各パラメータについて追加で2つの状態（モーメント）をメモリに保持する**。これが後述する「LoRAの方がGPUメモリが少なくて済む」理由の一つ（学習対象のパラメータが少ない＝オプティマイザーの状態も少なくて済む）。

### 損失（loss）/ 勾配（gradient）/ grad_norm
- **損失（loss）**: モデルの予測が正解からどれだけ外れているかを表す数値。今回はCausal LMの標準であるCross Entropy Loss。低いほど良い。
- **勾配（gradient）**: 損失を小さくするために各重みをどちらにどれだけ動かすべきかを示す値。逆伝播（backpropagation）で計算される。
- **grad_norm**: 全パラメータの勾配をまとめたベクトルの大きさ（L2ノルム）。学習ログに出てくる`grad_norm`はこれ。極端に大きい値が出ると学習が不安定な兆候。HF Trainerはデフォルトで`max_grad_norm=1.0`により勾配クリッピング（大きすぎる勾配を切り詰める）をしている。

### 過学習（overfitting）と検証データ（validation data）
学習データにだけ異常に適応してしまい、未知のデータに対する性能が落ちる現象。`--val_ratio`で学習データの一部を検証用に取り分け（学習には使わない）、`eval_loss`としてモニタリングすることで検知する。

実例（このプロジェクトのFull-FT実行結果）:

| epoch | train loss | eval loss |
|---|---|---|
| 1 | 1.29 | 1.29 |
| **2** | 0.85 | **1.26**（最小） |
| 3 | 0.65 | 1.30 |
| 5 | 0.30 | 1.42 |

train lossは下がり続けているのに、eval lossはepoch 2を境に上昇に転じている。これが過学習で、「epoch 2の時点のモデルが一番汎化性能が良い」ことを意味する。

---

## 2. PEFT / LoRA

### PEFTとは
**Parameter-Efficient Fine-Tuning**の略。モデル全体を学習し直す（Full Fine-Tuning）代わりに、ごく一部の追加パラメータだけを学習することで、少ないGPUメモリ・少ない学習時間で近い性能を狙う手法群の総称。LoRAはその代表的な一手法。`peft`はHugging Faceが提供するPEFT手法の実装ライブラリ名でもある。

### LoRAの仕組み（Low-Rank Adaptation）
ベースモデルのある重み行列 `W`（サイズ `d × d`）を直接更新する代わりに、`W`は凍結（学習しない）したまま、小さな2つの行列 `A`（`d × r`）と `B`（`r × d`）を新しく追加し、これだけを学習する。

```
出力 = W・x + (alpha / r) × B・(A・x)
        ^^^^^                ^^^^^^^^^^^^^^^^
        凍結（学習しない）      ここだけ学習する
```

`r`（ランク）はモデルの隠れ次元（数百〜数千）よりずっと小さい値（例: 8）にするため、`A`と`B`の合計パラメータ数は`W`単体よりずっと少ない。これが学習可能パラメータ数を劇的に減らせる理由。

実際にこのプロジェクトで確認した数値（`rinna/japanese-gpt2-medium`, `r=8`）:

| | 学習可能パラメータ | 割合 |
|---|---|---|
| Full-FT | 336,128,000 | 100% |
| LoRA | 2,162,688 | 0.64% |

### `main_lora.py`のLoRA関連引数

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--lora_r` | `8` | ランク`r`。大きいほど表現力は上がるが学習パラメータも増える |
| `--lora_alpha` | `16` | スケーリング係数。`alpha / r`が実質的な更新の強さになる。`alpha`を`r`の2倍にするのはPEFTの慣習的なデフォルト設定 |
| `--lora_dropout` | `0.05` | LoRA層に適用するdropout（過学習抑制） |
| `--lora_target_modules` | `c_attn,c_proj` | LoRAを挿入する層の名前。GPT-2は`nn.Linear`ではなく`Conv1D`という層でAttentionを実装しているが、`peft`は`Conv1D`にも対応しているのでそのまま指定できる |

### なぜLoRAは学習率を高めに設定するのか
Full-FTは大量のパラメータを少しずつ動かして良い解を探すのに対し、LoRAはごく少数のパラメータ（`A`, `B`）だけで元の重みの振る舞いを近似的に変化させる必要がある。動かせる自由度が少ない分、1回あたりの更新量を大きくしないと十分に適応しきれないため、経験的にLoRAはFull-FTより一桁程度高い学習率（`1e-4`〜`3e-4`程度）が使われることが多い。`main.py`（`5e-5`）と`main_lora.py`（`1e-4`）でデフォルト値をあえて変えているのはこのため。

---

## 3. Early Stopping（早期終了）

### 何を監視するか
毎エポック終了時に検証データで測った`eval_loss`を監視する。学習データの`loss`ではなく`eval_loss`を見るのがポイント（学習データのlossは基本的に下がり続けるので、過学習の検知には使えない）。

### patience（`--early_stopping_patience`）
「何回連続で改善が無ければ諦めるか」の回数。例えば`patience=3`なら、3エポック連続で`eval_loss`が過去の最小値を更新できなければ、その時点で学習を打ち切る。`0`を指定すると無効化され、`--epochs`で指定した回数まで必ず回す。

### `load_best_model_at_end` / `metric_for_best_model` / `greater_is_better`
`main.py`・`main_lora.py`内の`TrainingArguments`で使っている設定。

- `metric_for_best_model="eval_loss"`: 「どの指標で良し悪しを判断するか」を`eval_loss`に指定
- `greater_is_better=False`: `eval_loss`は小さいほど良いことを明示（精度(accuracy)のように大きいほど良い指標なら`True`にする）
- `load_best_model_at_end=True`: 学習が終わった時点のモデル（＝打ち切られた時点、過学習気味）ではなく、**学習中で`eval_loss`が最も低かった時点のモデル**を最終的に使うモデルとして復元する

この3つを組み合わせることで、「打ち切りはpatience分だけ様子を見つつ遅れて発生するが、採用されるのはあくまでベストな時点のモデル」という動きになる。学習ログの最後に出る

```
採用したモデル: output/full_ft\checkpoint-114（検証損失=1.2563）
```

がこれに対応する。`metrics.json`の`best_checkpoint`にも同じ情報が記録される（詳細は[README.md](../README.md)）。

学習中は内部的に一時チェックポイント（`checkpoint-*`）を作るが、学習終了後に最終モデルを1つだけ保存してから`checkpoint-*`は削除するようにしている（ストレージ節約のため）。

---

## 4. 用語 ⇔ コード引数 対応表

| 用語 | `main.py` / `main_lora.py`の引数・変数 |
|---|---|
| エポック | `--epochs` |
| バッチサイズ | `--batch_size` |
| 勾配累積 | `--grad_accum` |
| 学習率 | `--lr` |
| オプティマイザー | （固定）Trainerのデフォルト = AdamW |
| 検証データの割合 | `--val_ratio` |
| Early Stoppingのpatience | `--early_stopping_patience` |
| LoRAのランク | `--lora_r`（`main_lora.py`のみ） |
| LoRAのスケーリング | `--lora_alpha`（`main_lora.py`のみ） |
| LoRAを適用する層 | `--lora_target_modules`（`main_lora.py`のみ） |
| 学習可能パラメータ数 | `metrics.json`の`trainable_params` |
| ピークGPUメモリ | `metrics.json`の`peak_gpu_memory_gb` |
| 採用されたモデルの時点 | `metrics.json`の`best_checkpoint` |
