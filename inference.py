"""Production inference pipeline: single image or batch with ensemble + TTA."""
import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)
TTA_N = 10
CLASS_NAMES_PATH = "models/class_names.json"
FOLD_PATTERN = "models/fold_{}.keras"

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

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

print(f"Classes ({len(class_names)}): {class_names}")

# Load ensemble
fold_models = []
for fold in range(5):
    path = FOLD_PATTERN.format(fold)
    if os.path.exists(path):
        m = keras.models.load_model(path, custom_objects={"loss_fn": focal_loss()})
        fold_models.append(m)
        print(f"Loaded: {path}")
    else:
        fallback = ["models/fine_tuned_best.keras", "models/best_model.keras"]
        for fb in fallback:
            if os.path.exists(fb):
                m = keras.models.load_model(fb)
                fold_models = [m]
                print(f"Fallback (no ensemble): loaded {fb}")
                break
        break

if not fold_models:
    print("No models found!")
    exit(1)

# TTA augmentation
tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1),
])

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img

def predict_single(image_path):
    img = load_and_preprocess(image_path)

    # Standard prediction
    std_preds = []
    for m in fold_models:
        p = m.predict(tf.expand_dims(img, 0), verbose=0)
        std_preds.append(p)
    std_probs = tf.nn.softmax(np.mean(std_preds, axis=0)).numpy()[0]

    # TTA prediction
    tta_preds = []
    for _ in range(TTA_N):
        aug = tta_aug(tf.expand_dims(img, 0), training=True)
        aug = tf.clip_by_value(aug, 0, 255)
        ensemble_aug = []
        for m in fold_models:
            p = m.predict(aug, verbose=0)
            ensemble_aug.append(p)
        tta_preds.append(np.mean(ensemble_aug, axis=0))
    tta_probs = tf.nn.softmax(np.mean(tta_preds, axis=0)).numpy()[0]

    std_class = np.argmax(std_probs)
    tta_class = np.argmax(tta_probs)

    result = {
        "file": os.path.basename(image_path),
        "standard": {"class": class_names[std_class], "confidence": float(std_probs[std_class]), "all_probs": {c: float(std_probs[i]) for i, c in enumerate(class_names)}},
        "tta": {"class": class_names[tta_class], "confidence": float(tta_probs[tta_class]), "all_probs": {c: float(tta_probs[i]) for i, c in enumerate(class_names)}},
    }
    return result

def predict_batch(image_dir, output_json=None):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    if not paths:
        print(f"No images found in {image_dir}")
        return
    print(f"\nProcessing {len(paths)} images from {image_dir}...")
    results = []
    for path in paths:
        r = predict_single(path)
        results.append(r)
        print(f"  {r['file']:<30} Std: {r['standard']['class']:<12} ({r['standard']['confidence']:.2%})  "
              f"TTA: {r['tta']['class']:<12} ({r['tta']['confidence']:.2%})")
    if output_json:
        import json
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")

def predict_from_args():
    import argparse
    parser = argparse.ArgumentParser(description="XPawSure inference with ensemble + TTA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Path to a single image file")
    group.add_argument("--dir", "-d", help="Directory of images to batch process")
    parser.add_argument("--output", "-o", help="Output JSON file for batch results")
    args = parser.parse_args()

    if args.image:
        r = predict_single(args.image)
        print(f"\nResults for {r['file']}:")
        print(f"  Standard: {r['standard']['class']} ({r['standard']['confidence']:.2%})")
        print(f"  TTA x{TTA_N}: {r['tta']['class']} ({r['tta']['confidence']:.2%})")
        print("\n  Standard probabilities:")
        for c, p in r['standard']['all_probs'].items():
            print(f"    {c:<12} {p:.2%}")
        print(f"\n  TTA probabilities:")
        for c, p in r['tta']['all_probs'].items():
            print(f"    {c:<12} {p:.2%}")
    elif args.dir:
        predict_batch(args.dir, args.output)

if __name__ == "__main__":
    predict_from_args()