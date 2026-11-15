"""Generate the 4-section test-set evaluation for the production 5-class model.

Sections (per reviewer request):
  1. Test dataset     - image counts per condition
  2. Actual vs pred   - sample images with reference vs model prediction
  3. Confusion matrix - fresh 5x5 counts + row percentages
  4. Metrics          - accuracy, precision, recall, F1 (per-class + averages)

Read-only w.r.t. models/ and the dataset; writes only docs/EVALUATION/.
Model: models/fine_tuned_best_effnet_clean_5class_s42.keras (93.71%, seed 42).

Run:  .\\venv\\Scripts\\python.exe compare\\make_evaluation_report.py
"""
import os, json, csv, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = "models/fine_tuned_best_effnet_clean_5class_s42.keras"
CLASS_NAMES_PATH = "models/class_names_clean_5class_s42.json"
TEST_PATH = "ProcessedDataset224_Clean_5class/Test"
OUT_DIR = "docs/EVALUATION"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SAMPLES_PER_CLASS = 3
CLASS_WEIGHTS = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}

os.makedirs(OUT_DIR, exist_ok=True)


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=" * 64)
print("Production model test-set evaluation (read-only)")
print("=" * 64)

# ---------- load model ----------
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)
alpha = tf.constant([CLASS_WEIGHTS[i] for i in range(len(class_names))], dtype=tf.float32)
print(f"Loading model: {MODEL_PATH}")
try:
    model = keras.models.load_model(
        MODEL_PATH, custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
    )
except Exception as exc:
    print(f"  custom_objects failed ({exc}); retrying compile=False")
    model = keras.models.load_model(MODEL_PATH, compile=False)

# ---------- ordered file list (same order as image_dataset_from_directory) ----------
files = []
for cls in class_names:
    d = os.path.join(TEST_PATH, cls)
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")):
            files.append(os.path.join(d, fn))

test_ds = keras.utils.image_dataset_from_directory(
    TEST_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
assert list(test_ds.class_names) == class_names, "class order mismatch"

y_true, y_pred, y_conf, imgs = [], [], [], []
for batch in test_ds:
    logits = model.predict(batch[0], verbose=0)
    probs = tf.nn.softmax(logits).numpy()
    y_true.extend(batch[1].numpy())
    y_pred.extend(np.argmax(probs, axis=1))
    y_conf.extend(np.max(probs, axis=1))
    imgs.append(batch[0].numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_conf = np.array(y_conf)
imgs = np.concatenate(imgs, axis=0)
assert len(files) == len(y_true) == 143, f"expected 143, got files={len(files)} labels={len(y_true)}"

correct = y_true == y_pred
acc = float(correct.mean())
report = classification_report(
    y_true, y_pred, labels=list(range(len(class_names))),
    target_names=class_names, output_dict=True, zero_division=0,
)
report_txt = classification_report(
    y_true, y_pred, labels=list(range(len(class_names))),
    target_names=class_names, zero_division=0,
)
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

print(f"Test accuracy: {acc:.4f} ({correct.sum()}/{len(y_true)})")

# ---------- 1. class counts ----------
counts = {c: int((y_true == i).sum()) for i, c in enumerate(class_names)}
print("Counts:", counts)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_names, [counts[c] for c in class_names], color="#4C72B0")
ax.bar_label(bars, padding=2)
ax.set_ylabel("Number of test images")
ax.set_title(f"Test Set Images per Condition (total = {len(y_true)})")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "test_class_counts.png"), dpi=150)
plt.close()

# ---------- 2. sample predictions (errors first, then lowest-confidence correct) ----------
fig, axes = plt.subplots(len(class_names), SAMPLES_PER_CLASS, figsize=(4 * SAMPLES_PER_CLASS, 4.2 * len(class_names)))
for r, cls in enumerate(class_names):
    idxs = np.where(y_true == r)[0]
    wrong = sorted(idxs[~correct[idxs]], key=lambda i: -y_conf[i])
    right = sorted(idxs[correct[idxs]], key=lambda i: y_conf[i])
    picks = (wrong + right)[:SAMPLES_PER_CLASS]
    for c in range(SAMPLES_PER_CLASS):
        ax = axes[r][c]
        ax.axis("off")
        if c < len(picks):
            i = picks[c]
            ax.imshow(imgs[i].astype(np.uint8))
            status = "OK" if correct[i] else "WRONG"
            ax.set_title(
                f"actual: {class_names[y_true[i]]}\n"
                f"pred: {class_names[y_pred[i]]} ({y_conf[i]*100:.0f}%) [{status}]",
                fontsize=9,
                color=("#2a7" if correct[i] else "#c33"),
            )
fig.suptitle("Actual vs Predicted — sample test images (3 per condition; errors shown first)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUT_DIR, "sample_predictions.png"), dpi=130)
plt.close()

# ---------- 3. confusion matrix (counts + row %) ----------
fig, ax = plt.subplots(figsize=(8, 7))
row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
annot = np.array([[f"{cm[r, c]}\n({row_pct[r, c]:.0f}%)" for c in range(len(class_names))] for r in range(len(class_names))])
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)), class_names, rotation=20, ha="right")
ax.set_yticks(range(len(class_names)), class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — production model (accuracy {acc*100:.2f}%)")
for r in range(len(class_names)):
    for c in range(len(class_names)):
        ax.text(c, r, annot[r, c], ha="center", va="center",
                color="white" if cm[r, c] > cm.max() * 0.55 else "black", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ---------- 4. metrics csv + txt ----------
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_txt)

with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "precision", "recall", "f1", "support"])
    for cls in class_names:
        w.writerow([cls, round(report[cls]["precision"], 4), round(report[cls]["recall"], 4),
                    round(report[cls]["f1-score"], 4), int(report[cls]["support"])])
    w.writerow(["accuracy", "", "", round(acc, 4), len(y_true)])
    w.writerow(["macro avg", round(report["macro avg"]["precision"], 4),
                round(report["macro avg"]["recall"], 4),
                round(report["macro avg"]["f1-score"], 4), len(y_true)])
    w.writerow(["weighted avg", round(report["weighted avg"]["precision"], 4),
                round(report["weighted avg"]["recall"], 4),
                round(report["weighted avg"]["f1-score"], 4), len(y_true)])

# ---------- markdown ----------
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


counts_rows = [[c, counts[c], f"{counts[c]/len(y_true)*100:.1f}%"] for c in class_names]
counts_rows.append(["**Total**", f"**{len(y_true)}**", "**100%**"])

metric_rows = []
for cls in class_names:
    metric_rows.append([cls, f"{report[cls]['precision']:.3f}", f"{report[cls]['recall']:.3f}",
                        f"{report[cls]['f1-score']:.3f}", int(report[cls]['support'])])
metric_rows.append(["**Accuracy**", "", "", f"**{acc:.4f}**", len(y_true)])
metric_rows.append(["Macro avg", f"{report['macro avg']['precision']:.3f}",
                    f"{report['macro avg']['recall']:.3f}",
                    f"{report['macro avg']['f1-score']:.3f}", len(y_true)])
metric_rows.append(["Weighted avg", f"{report['weighted avg']['precision']:.3f}",
                    f"{report['weighted avg']['recall']:.3f}",
                    f"{report['weighted avg']['f1-score']:.3f}", len(y_true)])

cm_header = ["Actual \\ Predicted"] + class_names
cm_rows = [[f"**{c}**"] + [int(cm[i, j]) for j in range(len(class_names))]
           for i, c in enumerate(class_names)]
cm_md = md_table(cm_header, cm_rows)

md = f"""# Test-Set Evaluation — Production Model

- **Model:** `{MODEL_PATH}`
- **SHA-256:** `{sha256(MODEL_PATH)}`
- **Dataset:** `{TEST_PATH}` (5 conditions, {len(y_true)} images)
- **Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Overall accuracy:** **{acc*100:.2f}%** ({int(correct.sum())}/{len(y_true)} correct)

---

## 1. Test dataset

Number of images used for each of the five conditions:

{md_table(["Condition", "Images", "Share"], counts_rows)}

![Test class counts](test_class_counts.png)

---

## 2. Actual vs. predicted results

Sample images alongside their reference diagnoses and model predictions.
Each caption shows the reference (actual) diagnosis, the model's prediction,
its confidence, and whether the prediction was correct. Misclassified samples
are shown first. (3 samples per condition.)

![Actual vs predicted](sample_predictions.png)

---

## 3. Confusion matrix

How often each condition is correctly or incorrectly classified.
Cells show image counts and row percentages (row = actual condition).

![Confusion matrix](confusion_matrix.png)

{cm_md}

---

## 4. Classification metrics

Accuracy, precision, recall, and F1-score per condition and overall.

{md_table(["Class", "Precision", "Recall", "F1-score", "Support"], metric_rows)}

Full sklearn report: [`classification_report.txt`](classification_report.txt) · raw values: [`metrics.csv`](metrics.csv)
"""

with open(os.path.join(OUT_DIR, "TEST_EVALUATION.md"), "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nWritten to {OUT_DIR}/:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn}")
