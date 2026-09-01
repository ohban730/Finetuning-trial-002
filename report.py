import argparse
import csv
import json
import os

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"  # 日本語ラベルがtofu(□)化しないように指定
plt.rcParams["axes.unicode_minus"] = False


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_curves(log_history: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    train_points = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_points = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    return train_points, eval_points


def build_table(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for name, metrics in runs.items():
        peak_mem = metrics.get("peak_gpu_memory_gb")
        best = metrics.get("best_checkpoint")
        rows.append(
            {
                "run": name,
                "trainable_params": metrics["trainable_params"],
                "total_params": metrics["total_params"],
                "trainable_ratio_%": round(100 * metrics["trainable_params"] / metrics["total_params"], 4),
                "peak_gpu_memory_gb": round(peak_mem, 2) if peak_mem is not None else "N/A",
                "train_runtime_sec": round(metrics["train_runtime_sec"], 1),
                "final_train_loss": round(metrics["final_loss"], 4),
                "adopted_epoch": round(best["epoch"], 2) if best else "N/A",
                "adopted_eval_loss": round(best["eval_loss"], 4) if best else "N/A",
            }
        )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: list[dict], path: str) -> None:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_loss_curves(runs: dict[str, dict], path: str) -> None:
    plt.figure(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, metrics) in enumerate(runs.items()):
        train_points, eval_points = extract_curves(metrics["log_history"])
        color = colors[i % len(colors)]
        if train_points:
            xs, ys = zip(*train_points)
            plt.plot(xs, ys, label=f"{name} (train)", color=color, linestyle="-")
        if eval_points:
            xs, ys = zip(*eval_points)
            plt.plot(xs, ys, label=f"{name} (eval)", color=color, linestyle="--", marker="o")

        best = metrics.get("best_checkpoint")
        if best and best.get("epoch") is not None:
            plt.scatter(
                [best["epoch"]],
                [best["eval_loss"]],
                color=color,
                marker="*",
                s=300,
                zorder=5,
                edgecolors="black",
                label=f"{name} (採用モデル)",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="表示名:metrics.jsonのパス（例: Full-FT:output/full_ft/metrics.json）。複数指定可",
    )
    parser.add_argument("--output_dir", default="report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = {}
    for spec in args.run:
        name, path = spec.split(":", 1)
        runs[name] = load_metrics(path)

    rows = build_table(runs)
    write_csv(rows, os.path.join(args.output_dir, "comparison_table.csv"))
    write_markdown_table(rows, os.path.join(args.output_dir, "comparison_table.md"))
    plot_loss_curves(runs, os.path.join(args.output_dir, "loss_curve.png"))

    print(f"表とグラフを {args.output_dir} に保存しました。")


if __name__ == "__main__":
    main()
