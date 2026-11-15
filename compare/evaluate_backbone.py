"""Unified read-only evaluator: computes the full metric suite for any saved
checkpoint on the 5-class test set. Writes results under compare/results/.

Used to (re)evaluate the existing production EfficientNetB0 checkpoints and
any comparison checkpoints.
"""
import os, sys, json, argparse, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to .keras checkpoint")
parser.add_argument("--tag", type=str, required=True, help="Output tag, e.g. effnet_s42")
parser.add_argument("--label", type=str, default=None, help="Display name")
parser.add_argument("--test_dir", type=str, default="ProcessedDataset224_Clean_5class/Test",
                    help="Directory with class subdirs to evaluate on")
args = parser.parse_args()

DATASET_PATH = args.test_dir
IMAGE_SIZE = (224, 224)
RES_DIR = os.path.join(ROOT, "compare", "results")
os.makedirs(RES_DIR, exist_ok=True)

w5 = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}  # AD, Bact, Fung, Hot, Mange
w4 = {0: 2.0, 1: 2.5, 2: 1.0, 3: 1.3}  # Bact, Fung, Hot, Mange (alphabetical)


def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        p_t = tf.gather_nd(probs, tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1))
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if alpha is not None:
            alpha_t = tf.gather(alpha, tf.cast(y_true, tf.int32))
            focal = alpha_t * focal
        return tf.reduce_mean(focal)
    return loss_fn


test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, image_size=IMAGE_SIZE, batch_size=16, shuffle=False,
)
class_names = test_ds.class_names
assert len(class_names) in (4, 5), f"Unexpected class count: {len(class_names)}"
custom_weights = w5 if len(class_names) == 5 else w4
alpha = tf.constant([custom_weights[i] for i in range(len(class_names))], dtype=tf.float32)

print(f"Loading model: {args.model}")
model = tf.keras.models.load_model(
    args.model,
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())},
)

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

labels_idx = list(range(len(class_names)))
report = classification_report(
    y_true, y_pred, labels=labels_idx, target_names=class_names,
    output_dict=True, zero_division=0,
)
cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

# inference latency (per image, CPU), 50 images
lat = []
single_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, image_size=IMAGE_SIZE, batch_size=1, shuffle=False,
).take(50)
for images, _ in single_ds:
    t = time.time()
    model.predict(images, verbose=0)
    lat.append((time.time() - t) * 1000)
lat = np.array(lat)

# parse seed from tag like effnet_s42 or local_effnet_s42_on_public
import re
seed = None
m = re.search(r"_s(\d+)", args.tag)
if m:
    seed = int(m.group(1))

metrics = {
    "backbone": args.tag.split("_s")[0],
    "label": args.label or args.tag,
    "seed": seed,
    "stage": "finetune",
    "dataset": DATASET_PATH,
    "class_names": class_names,
    "test_accuracy": round(test_acc, 4),
    "macro_precision": round(report["macro avg"]["precision"], 4),
    "macro_recall": round(report["macro avg"]["recall"], 4),
    "macro_f1": round(report["macro avg"]["f1-score"], 4),
    "weighted_precision": round(report["weighted avg"]["precision"], 4),
    "weighted_recall": round(report["weighted avg"]["recall"], 4),
    "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    "per_class": {
        c: {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
        } for c in class_names
    },
    "confusion_matrix": cm.tolist(),
    "params_total": int(model.count_params()),
    "model_size_bytes": os.path.getsize(args.model),
    "inference_ms_mean": round(float(lat.mean()), 3),
    "inference_ms_std": round(float(lat.std()), 3),
    "checkpoint": args.model,
    "note": f"unified evaluator on {DATASET_PATH}",
}

out_json = os.path.join(RES_DIR, f"{args.tag}_metrics.json")
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(RES_DIR, f"{args.tag}_classification_report.txt"), "w") as f:
    f.write(classification_report(
        y_true, y_pred, labels=labels_idx, target_names=class_names, zero_division=0,
    ))

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {metrics['label']}")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, f"{args.tag}_cm.png"))
plt.close()

print(f"\nTest Accuracy: {test_acc:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
print(classification_report(
    y_true, y_pred, labels=labels_idx, target_names=class_names, zero_division=0,
))
print(f"Saved: {out_json}")
