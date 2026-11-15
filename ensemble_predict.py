"""Ensemble + TTA prediction and evaluation."""
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

def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        p_t = tf.gather_nd(probs, tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1))
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if alpha is not None:
            alpha_t = tf.gather(tf.constant(alpha, dtype=tf.float32), tf.cast(y_true, tf.int32))
            focal = alpha_t * focal
        return tf.reduce_mean(focal)
    return loss_fn

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
class_names = test_ds.class_names

with open("models/class_names.json") as f:
    saved_names = json.load(f)
if saved_names != class_names:
    print(f"Warning: class_names mismatch. Using: {class_names}")

print(f"\nClasses: {class_names}")
total_test = sum(len(files) for _, _, files in os.walk(f"{DATASET_PATH}/Test"))
print(f"Test samples: {total_test}")

# Load all fold models
fold_models = []
for fold in range(5):
    path = f"models/fold_{fold}.keras"
    if os.path.exists(path):
        m = keras.models.load_model(path, custom_objects={"loss_fn": focal_loss()})
        fold_models.append(m)
        print(f"Loaded: {path}")

if len(fold_models) == 0:
    print("No ensemble models found! Run ensemble_train.py first.")
    exit(1)

print(f"\nEnsemble size: {len(fold_models)} models")

# Compute weights based on validation accuracy
print("\nComputing model weights via validation accuracy...")
valid_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Valid", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
valid_ds = valid_ds.prefetch(AUTOTUNE)

model_weights = []
for i, m in enumerate(fold_models):
    _, val_acc = m.evaluate(valid_ds, verbose=0)
    model_weights.append(val_acc)
    print(f"  Model fold_{i}: val_acc = {val_acc:.4f}")

model_weights = np.array(model_weights)
model_weights = model_weights / model_weights.sum()  # normalize
print(f"\n  Normalized weights: {np.round(model_weights, 4)}")

# TTA augmentation
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

print(f"\nRunning ensemble + TTA ({TTA_N} augs) on {len(fold_models)} models...")
y_true = []
y_pred_std = []
y_pred_ensemble = []
y_pred_ensemble_tta = []

for images, labels in test_ds:
    imgs_np = images.numpy()
    y_true.extend(labels.numpy())

    # Standard single model
    preds_std = fold_models[0].predict(images, verbose=0)
    y_pred_std.extend(np.argmax(preds_std, axis=1))

    # Ensemble (no TTA) - weighted average logits
    ensemble_logits = []
    for m in fold_models:
        p = m.predict(images, verbose=0)
        ensemble_logits.append(p)
    avg_logits = np.average(ensemble_logits, axis=0, weights=model_weights)
    y_pred_ensemble.extend(np.argmax(avg_logits, axis=1))

    # Ensemble + TTA
    ensemble_tta_logits = []
    for m in fold_models:
        p = tta_predict(images, m, n=TTA_N)
        ensemble_tta_logits.append(p)
    avg_tta_logits = np.average(ensemble_tta_logits, axis=0, weights=model_weights)
    y_pred_ensemble_tta.extend(np.argmax(avg_tta_logits, axis=1))

y_true = np.array(y_true)
y_pred_std = np.array(y_pred_std)
y_pred_ensemble = np.array(y_pred_ensemble)
y_pred_ensemble_tta = np.array(y_pred_ensemble_tta)

std_acc = np.mean(y_true == y_pred_std)
ens_acc = np.mean(y_true == y_pred_ensemble)
ens_tta_acc = np.mean(y_true == y_pred_ensemble_tta)

print("\n" + "=" * 60)
print("Results Comparison")
print("=" * 60)
print(f"Single model (fold_0):            {std_acc:.4f}")
print(f"Ensemble ({len(fold_models)} models):             {ens_acc:.4f}  ({ens_acc - std_acc:+.4f})")
print(f"Ensemble + TTA x{TTA_N}:           {ens_tta_acc:.4f}  ({ens_tta_acc - std_acc:+.4f})")

print("\n=== Standard Classification Report ===")
print(classification_report(y_true, y_pred_std, target_names=class_names))

print("\n=== Ensemble Classification Report ===")
print(classification_report(y_true, y_pred_ensemble, target_names=class_names))

print("\n=== Ensemble + TTA Classification Report ===")
print(classification_report(y_true, y_pred_ensemble_tta, target_names=class_names))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, y_pred, title, acc, cmap in [
    (axes[0], y_pred_std, f"Single Model (Acc: {std_acc:.2%})", std_acc, "Blues"),
    (axes[1], y_pred_ensemble, f"Ensemble (Acc: {ens_acc:.2%})", ens_acc, "Greens"),
    (axes[2], y_pred_ensemble_tta, f"Ensemble+TTA x{TTA_N} (Acc: {ens_tta_acc:.2%})", ens_tta_acc, "Oranges"),
]:
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names).plot(
        cmap=cmap, ax=ax, colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.savefig("models/ensemble_comparison.png")
plt.close()
print("\nSaved: models/ensemble_comparison.png")

# Summary per class
print("\n=== Per-Class Accuracy ===")
print(f"{'Class':<12} {'Single':>10} {'Ensemble':>10} {'Ens+TTA':>10}")
for i, cls in enumerate(class_names):
    mask = y_true == i
    single = np.mean(y_pred_std[mask] == i) if mask.sum() > 0 else 0
    ens = np.mean(y_pred_ensemble[mask] == i) if mask.sum() > 0 else 0
    ens_tta = np.mean(y_pred_ensemble_tta[mask] == i) if mask.sum() > 0 else 0
    print(f"{cls:<12} {single:>8.2%} {ens:>8.2%} {ens_tta:>8.2%}")

print(f"\n{'Overall':<12} {std_acc:>8.2%} {ens_acc:>8.2%} {ens_tta_acc:>8.2%}")
print(f"{'Improvement':<12} {'':>10} {ens_acc - std_acc:>+8.2%} {ens_tta_acc - std_acc:>+8.2%}")