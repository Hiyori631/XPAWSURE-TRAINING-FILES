"""Aggregate all compare/results/*_metrics.json into:
  - docs/MODEL_COMPARISON/MODEL_COMPARISON_RESULTS.csv (per-run + mean/std rows)
  - docs/MODEL_COMPARISON/ comparative tables (markdown)
  - docs/MODEL_COMPARISON/ visualizations (bar charts, per-class recall, CMs avg)
Run after all 12 runs are complete.
"""
import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "compare", "results")
OUT = os.path.join(ROOT, "docs", "MODEL_COMPARISON")
os.makedirs(OUT, exist_ok=True)

ORDER = ["EfficientNetB0", "MobileNetV2", "ResNet50", "DenseNet121"]
COLORS = {"EfficientNetB0": "#1f77b4", "MobileNetV2": "#ff7f0e",
          "ResNet50": "#2ca02c", "DenseNet121": "#d62728"}
SEEDS = [42, 7, 123]

runs = []
for p in sorted(glob.glob(os.path.join(RES, "*_metrics.json"))):
    with open(p) as f:
        m = json.load(f)
    if m.get("stage") != "finetune":
        continue
    runs.append(m)

by_model = {}
for m in runs:
    by_model.setdefault(m["label"], []).append(m)

missing = [b for b in ORDER if b not in by_model]
if missing:
    print(f"WARNING: missing models: {missing}")

# ---------- CSV ----------
csv_path = os.path.join(OUT, "MODEL_COMPARISON_RESULTS.csv")
fields = ["model", "seed", "test_accuracy", "macro_precision", "macro_recall",
          "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
          "params_total", "model_size_mb", "duration_frozen_min", "duration_finetune_min",
          "inference_ms_mean", "inference_ms_std", "checkpoint"]
per_class_cols = None
rows = []
for label in ORDER:
    for m in sorted(by_model.get(label, []), key=lambda x: SEEDS.index(x["seed"]) if x.get("seed") in SEEDS else 99):
        pc = m["per_class"]
        if per_class_cols is None:
            per_class_cols = []
            for c in m["class_names"]:
                per_class_cols += [f"{c}_precision", f"{c}_recall", f"{c}_f1"]
        row = {
            "model": label,
            "seed": m.get("seed"),
            "test_accuracy": m["test_accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_precision": m["weighted_precision"],
            "weighted_recall": m["weighted_recall"],
            "weighted_f1": m["weighted_f1"],
            "params_total": m["params_total"],
            "model_size_mb": round(m["model_size_bytes"] / 1e6, 2),
            "duration_frozen_min": (round((m.get("duration_s_frozen") or 0) / 60, 1)
                                    if m.get("duration_s_frozen") else "N/A (existing checkpoint)"),
            "duration_finetune_min": (round((m.get("duration_s_finetune") or 0) / 60, 1)
                                      if m.get("duration_s_finetune") else "N/A (existing checkpoint)"),
            "inference_ms_mean": m.get("inference_ms_mean"),
            "inference_ms_std": m.get("inference_ms_std"),
            "checkpoint": m.get("checkpoint", ""),
        }
        for c in m["class_names"]:
            row[f"{c}_precision"] = pc[c]["precision"]
            row[f"{c}_recall"] = pc[c]["recall"]
            row[f"{c}_f1"] = pc[c]["f1"]
        rows.append(row)

all_fields = fields + (per_class_cols or [])

# mean / std rows per model
def agg_rows(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

summary = {}
for label in ORDER:
    ms = by_model.get(label, [])
    if not ms:
        continue
    s = {}
    for k in ["test_accuracy", "macro_precision", "macro_recall", "macro_f1",
              "weighted_f1", "inference_ms_mean"]:
        vals = [m[k] for m in ms if m.get(k) is not None]
        if vals:
            s[k] = agg_rows(vals)
    s["n"] = len(ms)
    s["params_total"] = ms[0]["params_total"]
    s["model_size_mb"] = round(ms[0]["model_size_bytes"] / 1e6, 2)
    s["durations"] = [ (m.get("duration_s_frozen_min") or ((m.get("duration_s_frozen") or 0)/60),
                        (m.get("duration_s_finetune_min") or ((m.get("duration_s_finetune") or 0)/60)) ) for m in ms ]
    s["per_class_recall"] = {}
    for c in ms[0]["class_names"]:
        vals = [m["per_class"][c]["recall"] for m in ms]
        s["per_class_recall"][c] = agg_rows(vals)
    s["per_class_f1"] = {}
    for c in ms[0]["class_names"]:
        vals = [m["per_class"][c]["f1"] for m in ms]
        s["per_class_f1"][c] = agg_rows(vals)
    summary[label] = s
    mr = mean, std = s["test_accuracy"]
    row = {f: "" for f in all_fields}
    row.update({
        "model": f"{label} MEAN±STD (n={s['n']})",
        "test_accuracy": f"{s['test_accuracy'][0]:.4f} ± {s['test_accuracy'][1]:.4f}",
        "macro_precision": f"{s['macro_precision'][0]:.4f} ± {s['macro_precision'][1]:.4f}",
        "macro_recall": f"{s['macro_recall'][0]:.4f} ± {s['macro_recall'][1]:.4f}",
        "macro_f1": f"{s['macro_f1'][0]:.4f} ± {s['macro_f1'][1]:.4f}",
        "weighted_f1": f"{s['weighted_f1'][0]:.4f} ± {s['weighted_f1'][1]:.4f}",
        "params_total": s["params_total"],
        "model_size_mb": s["model_size_mb"],
        "inference_ms_mean": f"{s['inference_ms_mean'][0]:.2f} ± {s['inference_ms_mean'][1]:.2f}",
    })
    rows.append(row)

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Wrote {csv_path}")

# ---------- markdown tables ----------
md = []
md.append("# Model Comparison — Results Tables\n")
md.append("Generated by `compare/aggregate_results.py`. All values are test-set metrics on "
          "`ProcessedDataset224_Clean_5class` (143 images), mean ± sample std over seeds 42/7/123.\n")

md.append("## Overall performance (mean ± std over 3 seeds)\n")
md.append("| Model | Accuracy | Macro F1 | Weighted F1 | Macro Recall | Params | Size (MB) | Inference (ms/img) |")
md.append("|---|---|---|---|---|---:|---:|---:|")
for label in ORDER:
    s = summary.get(label)
    if not s:
        continue
    md.append(f"| **{label}** | {s['test_accuracy'][0]:.4f} ± {s['test_accuracy'][1]:.4f} "
              f"| {s['macro_f1'][0]:.4f} ± {s['macro_f1'][1]:.4f} "
              f"| {s['weighted_f1'][0]:.4f} ± {s['weighted_f1'][1]:.4f} "
              f"| {s['macro_recall'][0]:.4f} ± {s['macro_recall'][1]:.4f} "
              f"| {s['params_total']:,} | {s['model_size_mb']} | "
              f"{s['inference_ms_mean'][0]:.1f} ± {s['inference_ms_mean'][1]:.1f} |")
md.append("")

md.append("## Per-run results\n")
md.append("| Model | Seed | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Frozen (min) | Finetune (min) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for label in ORDER:
    for m in sorted(by_model.get(label, []), key=lambda x: SEEDS.index(x["seed"]) if x.get("seed") in SEEDS else 99):
        df = f"{(m['duration_s_frozen']/60):.1f}" if m.get("duration_s_frozen") else "N/A"
        dt = f"{(m['duration_s_finetune']/60):.1f}" if m.get("duration_s_finetune") else "N/A"
        md.append(f"| {label} | {m.get('seed')} | {m['test_accuracy']:.4f} | {m['macro_precision']:.4f} "
                  f"| {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
                  f"| {df} | {dt} |")
md.append("")

classes = runs[0]["class_names"] if runs else []
md.append("## Per-class recall (mean ± std)\n")
md.append("| Model | " + " | ".join(classes) + " |")
md.append("|---|" + "---|" * len(classes))
for label in ORDER:
    s = summary.get(label)
    if not s:
        continue
    cells = [f"{s['per_class_recall'][c][0]:.3f} ± {s['per_class_recall'][c][1]:.3f}" for c in classes]
    md.append(f"| {label} | " + " | ".join(cells) + " |")
md.append("")

md.append("## Per-class F1 (mean ± std)\n")
md.append("| Model | " + " | ".join(classes) + " |")
md.append("|---|" + "---|" * len(classes))
for label in ORDER:
    s = summary.get(label)
    if not s:
        continue
    cells = [f"{s['per_class_f1'][c][0]:.3f} ± {s['per_class_f1'][c][1]:.3f}" for c in classes]
    md.append(f"| {label} | " + " | ".join(cells) + " |")
md.append("")

md_path = os.path.join(OUT, "RESULTS_TABLES.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"Wrote {md_path}")

# ---------- visualizations ----------
# 1. accuracy + macro F1 bars with error bars
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labels_present = [b for b in ORDER if b in summary]
x = np.arange(len(labels_present))
for ax, key, title in [(axes[0], "test_accuracy", "Test Accuracy"),
                       (axes[1], "macro_f1", "Macro F1-score")]:
    means = [summary[b][key][0] for b in labels_present]
    stds = [summary[b][key][1] for b in labels_present]
    cols = [COLORS[b] for b in labels_present]
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=cols, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title + " (mean ± std, 3 seeds)")
    ax.grid(axis="y", alpha=0.3)
    for b, m_, s_ in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, m_ + 0.01, f"{m_:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_macroF1_comparison.png"), dpi=150)
plt.close()

# 2. per-class recall grouped bars
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.2
x = np.arange(len(classes))
for i, label in enumerate(labels_present):
    s = summary[label]
    means = [s["per_class_recall"][c][0] for c in classes]
    stds = [s["per_class_recall"][c][1] for c in classes]
    ax.bar(x + (i - len(labels_present)/2 + 0.5) * width, means, width,
           yerr=stds, capsize=3, label=label, color=COLORS[label], alpha=0.85, edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=15)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Recall")
ax.set_title("Per-class Recall by Architecture (mean ± std, 3 seeds)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "per_class_recall_comparison.png"), dpi=150)
plt.close()

# 3. mean confusion matrices
for label in labels_present:
    cms = np.array([m["confusion_matrix"] for m in by_model[label]], dtype=float)
    mean_cm = cms.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = mean_cm[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > mean_cm.max()/2 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Mean Confusion Matrix — {label} (3 seeds, counts)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    safe = label.lower().replace(" ", "")
    plt.savefig(os.path.join(OUT, f"cm_mean_{safe}.png"), dpi=150)
    plt.close()

# 4. training duration bars (models with recorded durations only)
fig, ax = plt.subplots(figsize=(9, 5))
dur_labels = [b for b in labels_present
              if all(sum(d) > 0 for d in summary[b]["durations"])]
frozen = [sum(d[0] for d in summary[b]["durations"]) / len(summary[b]["durations"]) for b in dur_labels]
finet = [sum(d[1] for d in summary[b]["durations"]) / len(summary[b]["durations"]) for b in dur_labels]
ax.bar(dur_labels, frozen, label="Frozen phase", color="#888", edgecolor="black")
ax.bar(dur_labels, finet, bottom=frozen, label="Fine-tune phase", color="#333", edgecolor="black")
ax.set_ylabel("Mean duration (min)")
ax.set_title("Mean training duration per model (EfficientNetB0: existing checkpoints, duration not recorded)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_duration_comparison.png"), dpi=150)
plt.close()

print(f"Visualizations written to {OUT}")

# machine-readable summary for the report
with open(os.path.join(OUT, "_summary.json"), "w") as f:
    json.dump({b: {k: (v if not isinstance(v, tuple) else list(v))
                   for k, v in s.items() if k != "durations"} for b, s in summary.items()},
              f, indent=2, default=str)
print("Done.")
