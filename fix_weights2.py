"""
Regenerate weights.bin directly from the Keras model's weights.
The issue: normalization/count is stored as float64 (8 bytes) but manifest says float32 (4 bytes).
This causes ALL subsequent weights to be offset by 4 bytes.

Fix: Extract all weights from the production model (with Softmax), cast to float32,
and write them in the order specified by the mobile manifest.
"""

import tensorflow as tf
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
MOBILE_MODEL_DIR = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model"
MOBILE_MODEL_PATH = os.path.join(MOBILE_MODEL_PATH := MOBILE_MODEL_DIR, "model.json")
OUT_BIN = os.path.join(MOBILE_MODEL_DIR, "weights_fixed.bin")

class_weights = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}
alpha = tf.constant([class_weights[i] for i in range(5)], dtype=tf.float32)
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

print("Loading Keras model...")
model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

# Build the production model (same as convert_to_tfjs.py)
inputs = model.input
outputs = tf.keras.layers.Softmax()(model.output)
prod_model = tf.keras.Model(inputs, outputs, name="xpawsure_model")

print("Production model weights:")
for i, w in enumerate(prod_model.weights):
    arr = w.numpy()
    print(f"  {i}: {w.name:60s} shape={str(arr.shape):20s} dtype={arr.dtype} nbytes={arr.nbytes}")

print(f"\nTotal weights: {len(prod_model.weights)}")

# Check: are any weights NOT float32?
non_f32 = []
for w in prod_model.weights:
    arr = w.numpy()
    if arr.dtype != np.float32:
        non_f32.append((w.name, arr.dtype))
        
if non_f32:
    print(f"\nNon-float32 weights found:")
    for name, dtype in non_f32:
        print(f"  {name}: {dtype}")
else:
    print("\nAll weights are float32")

# Read the mobile model manifest to get expected shapes
with open(MOBILE_MODEL_PATH, 'r') as f:
    mobile_json = json.load(f)

manifest_specs = mobile_json['weightsManifest'][0]['weights']

# Build a lookup from production model weights by matching shapes and position
# The production model weights should be in the SAME ORDER as the manifest
print("\nRegenerating weights_fixed.bin...")
with open(OUT_BIN, 'wb') as f:
    for i, w in enumerate(prod_model.weights):
        arr = w.numpy().astype(np.float32)
        f.write(arr.tobytes())
        print(f"  Wrote {i}: {w.name} shape={arr.shape} nbytes={arr.nbytes}")

new_size = os.path.getsize(OUT_BIN)
old_size = os.path.getsize(os.path.join(MOBILE_MODEL_DIR, "weights.bin"))
print(f"\nOld weights.bin: {old_size} bytes")
print(f"New weights_fixed.bin: {new_size} bytes")
print(f"Difference: {old_size - new_size} bytes")

# Now update the manifest to use these new weights
# But first verify: the new binary should have ALL float32 data
with open(OUT_BIN, 'rb') as f:
    new_raw = f.read()

# Quick check: first 40 bytes
new_floats = np.frombuffer(new_raw[:40], dtype=np.float32)
print(f"\nNew binary first 10 float32:")
for i, v in enumerate(new_floats[:10]):
    print(f"  [{i}] {v:.8f}")

# Check stem_conv kernel alignment
print("\nVerifying stem_conv kernel alignment in new binary...")
# Skip first 3 weights: normalization/mean(3) + normalization/variance(3) + normalization/count(1) = 7 floats = 28 bytes
offset = 28
kernel_data = np.frombuffer(new_raw[offset:offset+3*3*3*32*4], dtype=np.float32).reshape(3,3,3,32)
print(f"  stem_conv kernel first value: {kernel_data.flat[0]}")
print(f"  stem_conv kernel min: {kernel_data.min():.6f} max: {kernel_data.max():.6f}")
