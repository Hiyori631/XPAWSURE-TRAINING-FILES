"""
verify_5class_s42.py - Read-only re-evaluation of the SHIPPED 5-class model.

Purpose: settle whether the shipped model's standard test accuracy is
93.71% (AGENTS.md claim) or 88.81% (saved confusion matrix claim).

- Evaluates models/fine_tuned_best_effnet_clean_5class_s42.keras
  (the exact file convert_to_tfjs.py ships to mobile, line 24)
- On ProcessedDataset224_Clean_5class/Test (143 images)
- Replicates fine_tune_effnet.py's eval pipeline exactly
  (image_dataset_from_directory params, focal loss, same model load)
- Writes NOTHING to disk. Console output only.

Run:  .\\venv\\Scripts\\python.exe verify_5class_s42.py
"""

import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

MODEL_PATH = "models/fine_tuned_best_effnet_clean_5class_s42.keras"
CLASS_NAMES_PATH = "models/class_names_clean_5class_s42.json"
TEST_PATH = "ProcessedDataset224_Clean_5class/Test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
TOTAL_TEST_IMAGES = 143
# Alphabetical 5-class order (AGENTS.md line 104)
CLASS_WEIGHTS = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}


def focal_loss(gamma=2.0, alpha=None):
    # Identical to fine_tune_effnet.py
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        p_t = tf.gather_nd(
            probs,
            tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1),
        )
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if alpha is not None:
            alpha_t = tf.gather(alpha, tf.cast(y_true, tf.int32))
            focal = alpha_t * focal
        return tf.reduce_mean(focal)

    return loss_fn


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 64)
    print("XPawSure shipped-model verification (read-only, nothing written)")
    print("=" * 64)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"TensorFlow  : {tf.__version__}")
    print(f"CWD         : {os.getcwd()}")

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: model not found: {MODEL_PATH}")
        sys.exit(1)

    stat = os.stat(MODEL_PATH)
    print(f"Model       : {MODEL_PATH}")
    print(f"Model size  : {stat.st_size:,} bytes")
    print(f"Model mtime : {__import__('datetime').datetime.fromtimestamp(stat.st_mtime)}")
    print(f"Model SHA256: {file_sha256(MODEL_PATH)}")

    alpha = tf.constant(
        [CLASS_WEIGHTS[i] for i in range(len(CLASS_WEIGHTS))], dtype=tf.float32
    )

    print("\n[1/4] Loading model...")
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())},
        )
        print("      loaded with custom_objects (same as fine_tune_effnet.py)")
    except Exception as exc:
        print(f"      custom_objects failed ({type(exc).__name__}: {exc})")
        print("      retrying with compile=False...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("      loaded with compile=False (weights only, inference-equivalent)")

    print("\n[2/4] Loading test dataset...")
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    print(f"      classes from dataset : {test_ds.class_names}")
    print(f"      classes from json    : {class_names}")
    if list(test_ds.class_names) != list(class_names):
        print("FATAL: class order mismatch between dataset and class_names json")
        sys.exit(1)

    print("\n[3/4] Evaluating (model.evaluate)...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"      test_loss = {test_loss:.4f}")
    print(f"      test_acc  = {test_acc * 100:.4f}%")

    print("\n[4/4] Per-image predictions (report + confusion matrix)...")
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    preds = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    assert len(y_true) == TOTAL_TEST_IMAGES, f"expected {TOTAL_TEST_IMAGES}, got {len(y_true)}"

    correct = int(np.sum(y_true == y_pred))

    print("\n" + "=" * 64)
    print(f"RESULT: {correct}/{TOTAL_TEST_IMAGES} = {correct / TOTAL_TEST_IMAGES * 100:.2f}%")
    print("=" * 64)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=3, zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=true, cols=pred):")
    header = f"{'':>22}" + "".join(f"{c[:11]:>13}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i][:20]:>22}" + "".join(f"{v:>13}" for v in row))

    print("\nVERDICT:")
    if correct == 134:
        print("  134/143 -> matches AGENTS.md (93.71%).")
        print("  The saved confusion-matrix png is stale/mislabeled.")
        print("  Shipped model IS the 93.71% model. No model swap needed.")
    elif correct == 127:
        print("  127/143 -> matches the saved confusion matrix (88.81%).")
        print("  AGENTS.md's 93.71% claim is NOT reproducible from this model.")
        print("  Consider evaluating/switching to seed 123 (90.91%, artifact-verified).")
    else:
        print(f"  {correct}/143 matches NEITHER source (134 or 127).")
        print("  Recheck both sources; possible dataset or model drift.")

    print("\nNOTE: this script writes no files. Output above is the only artifact.")


if __name__ == "__main__":
    main()
