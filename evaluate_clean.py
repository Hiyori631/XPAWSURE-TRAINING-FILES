import os, json, numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

TEST_PATH = "ProcessedDataset224_Clean/Test"
image_paths = []
y_true = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(TEST_PATH, class_name)
    for fname in sorted(os.listdir(class_folder)):
        image_paths.append(os.path.join(class_folder, fname))
        y_true.append(class_index)

print(f"Test images: {len(image_paths)}")

batch_size = 32
all_logits = []
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch = []
    for p in batch_paths:
        img = keras.utils.load_img(p, target_size=(224, 224))
        img_array = keras.utils.img_to_array(img)
        batch.append(img_array)
    batch = np.array(batch)
    logits = model.predict(batch, verbose=0)
    all_logits.append(logits)

all_logits = np.concatenate(all_logits, axis=0)
probs = tf.nn.softmax(all_logits).numpy()
y_pred = np.argmax(probs, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"\n{'='*50}")
print(f"TEST ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*50}")

print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix (rows=true, cols=pred):")
header = "".join(f"{c:>10}" for c in class_names)
print(f"{'':>10}{header}")
for i, c in enumerate(class_names):
    row = "".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
    print(f"{c:>10}{row}")

print(f"\nPer-class accuracy:")
for i, c in enumerate(class_names):
    correct = cm[i,i]
    total = cm[i].sum()
    print(f"  {c:<12} {correct}/{total} = {correct/total*100:.2f}%")

print(f"\nMisclassified:")
mis = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_true[i]:
        mis += 1
        fname = os.path.basename(image_paths[i])
        true_name = class_names[y_true[i]]
        pred_name = class_names[y_pred[i]]
        conf = probs[i][y_pred[i]] * 100
        print(f"  {fname:35s} True: {true_name:12s} Pred: {pred_name:12s} ({conf:.2f}%)")
print(f"\nTotal misclassified: {mis}/{len(y_pred)}")
