"""Test-Time Augmentation (TTA) prediction and evaluation."""
import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATASET_PATH = "ProcessedDataset224_PublicLocal"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
TTA_N = 10
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("models", exist_ok=True)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
class_names = test_ds.class_names

with open("models/class_names.json") as f:
    saved_names = json.load(f)
if saved_names != class_names:
    print(f"Warning: class_names mismatch. Using: {class_names}")

print(f"\nClasses: {class_names}")
print(f"Test samples: {len(test_ds)} batches ({sum(len(files) for _, _, files in os.walk(f'{DATASET_PATH}/Test'))} total)")

model_paths = [
    "models/fine_tuned_best.keras",
    "models/fine_tuned_final.keras",
    "models/best_model.keras"
]

best_model_path = None
for p in model_paths:
    if os.path.exists(p):
        best_model_path = p
        break

if best_model_path is None:
    print("No model found! Train a model first.")
    exit(1)

print(f"\nLoading model: {best_model_path}")
model = keras.models.load_model(best_model_path)

# TTA augmentation pipeline
tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1),
])

def tta_predict(images, model, n=TTA_N):
    preds = []
    for _ in range(n):
        aug_images = tta_aug(images, training=True)
        aug_images = tf.clip_by_value(aug_images, 0, 255)
        p = model.predict(aug_images, verbose=0)
        preds.append(p)
    return np.mean(preds, axis=0)

print(f"\nRunning TTA with {TTA_N} augmentations per image...")
y_true, y_pred_std, y_pred_tta = [], [], []

for images, labels in test_ds:
    # Standard prediction (no TTA)
    preds_std = model.predict(images, verbose=0)
    # TTA prediction
    preds_tta = tta_predict(images, model, n=TTA_N)

    y_true.extend(labels.numpy())
    y_pred_std.extend(np.argmax(preds_std, axis=1))
    y_pred_tta.extend(np.argmax(preds_tta, axis=1))

# Standard metrics
std_acc = np.mean(np.array(y_true) == np.array(y_pred_std))
tta_acc = np.mean(np.array(y_true) == np.array(y_pred_tta))

print("\n" + "=" * 60)
print("TTA Results")
print("=" * 60)
print(f"Standard prediction accuracy: {std_acc:.4f}")
print(f"TTA prediction accuracy:     {tta_acc:.4f}")
print(f"Improvement:                 {tta_acc - std_acc:+.4f}")

print("\n\n=== Standard Classification Report ===")
print(classification_report(y_true, y_pred_std, target_names=class_names))

print("\n=== TTA Classification Report ===")
print(classification_report(y_true, y_pred_tta, target_names=class_names))

# Confusion matrices side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred_std), display_labels=class_names).plot(
    cmap="Blues", ax=ax1, colorbar=False)
ax1.set_title(f"Standard (Acc: {std_acc:.2%})")

ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred_tta), display_labels=class_names).plot(
    cmap="Blues", ax=ax2, colorbar=False)
ax2.set_title(f"TTA x{TTA_N} (Acc: {tta_acc:.2%})")

plt.savefig("models/tta_comparison.png")
plt.close()
print("\nSaved: models/tta_comparison.png")

# Per-class comparison
print("\n=== Per-Class Accuracy ===")
for i, cls in enumerate(class_names):
    mask = np.array(y_true) == i
    std_cls = np.mean(np.array(y_pred_std)[mask] == i) if mask.sum() > 0 else 0
    tta_cls = np.mean(np.array(y_pred_tta)[mask] == i) if mask.sum() > 0 else 0
    print(f"  {cls:<12}  Std: {std_cls:.2%}  TTA: {tta_cls:.2%}  {'+' if tta_cls >= std_cls else ''}{tta_cls - std_cls:+.2%}")

print(f"\nOverall improvement from TTA: {tta_acc - std_acc:+.2%}")