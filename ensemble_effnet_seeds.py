"""Ensemble of EffNet seeds: average softmax from multiple seed models."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SEEDS = [42, 7, 123]
DATASET_PATH = "ProcessedDataset224_PublicLocal"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

custom_weights = {0: 2.0, 1: 2.5, 2: 1.0, 3: 1.3}
alpha = tf.constant([custom_weights[i] for i in range(4)], dtype=tf.float32)

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
    f"{DATASET_PATH}/Test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
class_names = test_ds.class_names

print("Loading models...")
models = []
for seed in SEEDS:
    path = f"models/fine_tuned_best_effnet_s{seed}.keras"
    m = keras.models.load_model(path, custom_objects={"loss_fn": focal_loss(alpha=alpha.numpy())})
    models.append(m)
    print(f"  Seed {seed}: loaded")

all_preds = []
all_labels = []
for images, labels in test_ds:
    logits = [m.predict(images, verbose=0) for m in models]
    avg_logits = np.mean(logits, axis=0)
    all_preds.extend(np.argmax(avg_logits, axis=1))
    all_labels.extend(labels.numpy())

acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n{'='*60}")
print(f"Ensemble Test Accuracy (seeds {SEEDS}): {acc:.4f}")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, colorbar=False)
plt.title(f"Ensemble EffNet Seeds {SEEDS} — Acc={acc:.2%}")
plt.tight_layout()
plt.savefig("models/ensemble_effnet_cm.png")
print(f"\nConfusion matrix saved to models/ensemble_effnet_cm.png")

with open("models/ensemble_effnet_result.txt", "w") as f:
    f.write(f"Ensemble accuracy: {acc:.4f}\n")
    f.write(f"Seeds: {SEEDS}\n")
print("Result saved to models/ensemble_effnet_result.txt")
