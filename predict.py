from tensorflow import keras
import tensorflow as tf
import numpy as np
import json
from PIL import Image

TTA_N = 10
CONFIDENCE_THRESHOLD = 60.0

QUESTIONS = {
    (0, 1): ("Does the lesion have pus or a moist discharge?", 0),
    (0, 2): ("Did it appear suddenly and spread rapidly?", 2),
    (0, 3): ("Is there significant hair loss?", 3),
    (0, 4): ("Does the skin show hives, welts, or raised bumps?", 4),
    (1, 2): ("Is the lesion moist or weepy?", 2),
    (1, 3): ("Is the itching intense?", 3),
    (1, 4): ("Is the itching intense and widespread?", 4),
    (2, 3): ("Is the skin thickened or crusty?", 3),
    (2, 4): ("Did the rash appear after exposure to a new product or allergen?", 4),
    (3, 4): ("Are there raised bumps or welts rather than crusty patches?", 4),
}

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

class_weights = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}
alpha = tf.constant([class_weights[i] for i in range(5)], dtype=tf.float32)

model = keras.models.load_model(
    "models/fine_tuned_best_effnet_clean_5class_s42.keras",
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

with open("models/class_names_clean_5class_s42.json") as f:
    class_names = json.load(f)

tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1),
])

image_path = input("Enter image path: ").strip().strip('"')

print(class_names)

img = Image.open(image_path).convert("RGB")
img_array = np.array(img, dtype=np.float32)
img_resized = tf.image.resize(img_array, (224, 224))

logits = model.predict(tf.expand_dims(img_resized, 0), verbose=0)
std_probs = tf.nn.softmax(logits).numpy()[0]

tta_logits = []
for _ in range(TTA_N):
    aug = tta_aug(tf.expand_dims(img_resized, 0), training=True)
    aug = tf.clip_by_value(aug, 0, 255)
    l = model.predict(aug, verbose=0)
    tta_logits.append(l)
probs = tf.nn.softmax(np.mean(tta_logits, axis=0)).numpy()[0]

pred = np.argmax(probs)
conf = np.max(probs) * 100

print("\n==========================")
print("Prediction (with TTA)")
print("==========================")

if conf >= CONFIDENCE_THRESHOLD:
    print(f"Disease: {class_names[pred]}")
    print(f"Confidence: {conf:.2f}%")
else:
    print(f"Disease: Uncertain / Out-of-distribution")
    print(f"The model is not confident (only {conf:.2f}%).")
    print(f"Best guess: {class_names[pred]}")

    sorted_idx = np.argsort(probs)[::-1]
    top2 = sorted_idx[:2]
    pair_key = tuple(sorted(top2))
    pair_key_rev = (top2[1], top2[0])

    if pair_key in QUESTIONS:
        question, yes_class = QUESTIONS[pair_key]
        no_class = top2[0] if yes_class == top2[1] else top2[1]
        print(f"\nTo improve accuracy, please answer:")
        print(f"  {question}")
        ans = input("  (yes/no/?) > ").strip().lower()
        if ans == "yes":
            probs[yes_class] *= 2.0
            probs[no_class] *= 0.5
        elif ans == "no":
            probs[no_class] *= 2.0
            probs[yes_class] *= 0.5

        probs /= probs.sum()
        pred = np.argmax(probs)
        conf = np.max(probs) * 100

        print()
        if conf >= CONFIDENCE_THRESHOLD:
            print(f"→ Final: {class_names[pred]} ({conf:.2f}%)")
        else:
            print(f"→ Still uncertain — best guess: {class_names[pred]} ({conf:.2f}%)")
    else:
        print(f"\n  Top candidates: {class_names[top2[0]]} ({probs[top2[0]]*100:.2f}%), {class_names[top2[1]]} ({probs[top2[1]]*100:.2f}%)")

print("\nAll Probabilities:")
for disease, score in zip(class_names, probs):
    print(f"  {disease}: {score*100:.2f}%")
