import os, json, numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TTA_N = 10
CONFIDENCE_THRESHOLD = 60.0

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

class_weights = {0: 2.0, 1: 2.5, 2: 1.0, 3: 1.3}
alpha = tf.constant([class_weights[i] for i in range(4)], dtype=tf.float32)

model = keras.models.load_model(
    "models/fine_tuned_best_effnet_clean_s42.keras",
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

with open("models/class_names_clean_s42.json") as f:
    class_names = json.load(f)

tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1),
])

TEST_PATH = "ProcessedDataset224_Clean/Test"
image_paths = []
y_true = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(TEST_PATH, class_name)
    for fname in sorted(os.listdir(class_folder)):
        image_paths.append(os.path.join(class_folder, fname))
        y_true.append(class_index)

print(f"Test images: {len(image_paths)}")

std_probs_all = []
tta_probs_all = []
for i, p in enumerate(image_paths):
    img = tf.image.decode_image(tf.io.read_file(p), channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img_batch = tf.expand_dims(img, 0)

    std_logits = model.predict(img_batch, verbose=0)
    std_probs = tf.nn.softmax(std_logits).numpy()[0]

    tta_logits = []
    for _ in range(TTA_N):
        aug = tta_aug(img_batch, training=True)
        aug = tf.clip_by_value(aug, 0, 255)
        l = model.predict(aug, verbose=0)
        tta_logits.append(l)
    tta_probs = tf.nn.softmax(np.mean(tta_logits, axis=0)).numpy()[0]

    std_probs_all.append(std_probs)
    tta_probs_all.append(tta_probs)

    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(image_paths)}")

std_probs_all = np.array(std_probs_all)
tta_probs_all = np.array(tta_probs_all)

def report(name, probs, use_threshold=True):
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1) * 100

    all_acc = accuracy_score(y_true, y_pred)
    confident_mask = confs >= CONFIDENCE_THRESHOLD
    n_confident = confident_mask.sum()
    n_uncertain = len(y_true) - n_confident
    acc_confident = accuracy_score(np.array(y_true)[confident_mask], y_pred[confident_mask])

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Overall Accuracy (all 130): {all_acc*100:.2f}%")
    print(f"Confident predictions (>= {CONFIDENCE_THRESHOLD:.0f}%): {n_confident}/{len(y_true)}")
    print(f"Uncertain (< {CONFIDENCE_THRESHOLD:.0f}%): {n_uncertain}")
    if n_confident > 0:
        print(f"Accuracy among confident: {acc_confident*100:.2f}%")
    if n_uncertain > 0:
        uncertain_ground_truth = [class_names[y_true[i]] for i in range(len(y_true)) if not confident_mask[i]]
        from collections import Counter
        print(f"Uncertain distribution: {dict(Counter(uncertain_ground_truth))}")

    print(f"\nConfusion Matrix (all 130, rows=true):")
    cm = confusion_matrix(y_true, y_pred)
    header = "".join(f"{c:>10}" for c in class_names)
    print(f"{'':>10}{header}")
    for i, c in enumerate(class_names):
        row = "".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
        print(f"{c:>10}{row}")

    print(f"\nMisclassified by new flow:")
    mis_pred = y_pred[np.array(y_true) != y_pred]
    mis_true = [y_true[i] for i in range(len(y_true)) if y_pred[i] != y_true[i]]
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            flag = "UNCERTAIN" if not confident_mask[i] else ""
            fname = os.path.basename(image_paths[i])
            print(f"  {fname:35s} True: {class_names[y_true[i]]:12s} Pred: {class_names[y_pred[i]]:12s} Conf: {confs[i]:.1f}% {flag}")
    return

report("Standard (no TTA)", std_probs_all)
print("\n" + "=" * 50)
report("TTA (N=10)", tta_probs_all)