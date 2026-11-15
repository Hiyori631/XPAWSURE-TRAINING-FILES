"""
Fix the weights.bin alignment issue.
The normalization/count variable is stored as float64 (8 bytes) in the binary file,
but the manifest says float32 (4 bytes). This causes a 4-byte misalignment for
every subsequent weight.

This script regenerates weights.bin with all weights properly cast to float32.
"""

import tensorflow as tf
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

KERAS_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
MOBILE_MODEL_DIR = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model"

print("Loading Keras model...")
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

model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

# Read the mobile model.json to get the weight manifest
mobile_model_path = os.path.join(MOBILE_MODEL_DIR, "model.json")
with open(mobile_model_path, 'r') as f:
    mobile_json = json.load(f)

weight_specs = mobile_json['weightsManifest'][0]['weights']

# Build a mapping from TF weight names to numpy arrays
# TF.js weight names: "layer_name/weight_name"
# Keras weight names: may have different prefix
keras_weight_map = {}
for w in model.weights:
    # Keras 3 weight name format: "layer_name/variable_name:0"
    name = w.name
    # Strip the ":0" suffix
    if name.endswith(':0'):
        name = name[:-2]
    keras_weight_map[name] = w.numpy()

print(f"Keras weight count: {len(keras_weight_map)}")
print(f"Mobile manifest weight count: {len(weight_specs)}")

# For each weight in the mobile manifest, find the matching Keras weight
# and write it as float32
print("\nRegenerating weights.bin with all weights as float32...")

new_weights_bin = os.path.join(MOBILE_MODEL_DIR, "weights_fixed.bin")
with open(new_weights_bin, 'wb') as f:
    matched = 0
    unmatched = 0
    for spec in weight_specs:
        name = spec['name']
        shape = spec['shape']

        if name in keras_weight_map:
            arr = keras_weight_map[name]
            # Cast to float32
            arr_f32 = arr.astype(np.float32)
            if list(arr_f32.shape) != shape:
                print(f"  WARNING: Shape mismatch for {name}: manifest={shape} actual={arr_f32.shape}")
            f.write(arr_f32.tobytes())
            matched += 1
        else:
            print(f"  UNMATCHED: {name}")
            # Try to find it by searching partial names
            found = False
            for kname, karr in keras_weight_map.items():
                if name.replace('/', '/') in kname or kname.endswith(name):
                    arr_f32 = karr.astype(np.float32)
                    f.write(arr_f32.tobytes())
                    print(f"    -> Matched as: {kname}")
                    matched += 1
                    found = True
                    break
            if not found:
                print(f"    -> SKIPPED (no match)")
                unmatched += 1

    print(f"\nMatched: {matched}, Unmatched: {unmatched}")

# Verify the new binary
print("\nVerifying new weights.bin...")
with open(new_weights_bin, 'rb') as f:
    new_raw = f.read()

with open(os.path.join(MOBILE_MODEL_DIR, "weights.bin"), 'rb') as f:
    old_raw = f.read()

print(f"Old binary size: {len(old_raw)} bytes")
print(f"New binary size: {len(new_raw)} bytes")
print(f"Different: {new_raw != old_raw}")

# Check first 40 bytes
new_floats = np.frombuffer(new_raw[:40], dtype=np.float32)
print(f"\nNew binary first 40 bytes as float32:")
for i, v in enumerate(new_floats):
    print(f"  byte offset {i*4:2d}: {v}")

# Quick verification: check stem_conv kernel
print("\nVerifying stem_conv kernel alignment...")
with open(mobile_model_path, 'r') as f:
    mobile_json2 = json.load(f)

offset = 0
for spec in mobile_json2['weightsManifest'][0]['weights']:
    n = 1
    for s in spec['shape']:
        n *= s
    if spec['name'] == 'stem_conv/kernel':
        new_kernel = np.frombuffer(new_raw[offset:offset+n*4], dtype=np.float32).reshape(spec['shape'])
        python_kernel = keras_weight_map.get('stem_conv/kernel') or keras_weight_map.get('efficientnetb0/stem_conv/kernel')
        if python_kernel is not None:
            diff = np.abs(new_kernel.flatten() - python_kernel.astype(np.float32).flatten())
            print(f"  Max diff: {diff.max():.10f}")
            print(f"  Mean diff: {diff.mean():.10f}")
            print(f"  Identical: {np.allclose(new_kernel, python_kernel, atol=1e-6)}")
        break
    offset += n * 4

# Also verify stem_bn
offset = 0
for spec in mobile_json2['weightsManifest'][0]['weights']:
    n = 1
    for s in spec['shape']:
        n *= s
    if spec['name'] == 'stem_bn/gamma':
        new_gamma = np.frombuffer(new_raw[offset:offset+n*4], dtype=np.float32).reshape(spec['shape'])
        python_gamma = keras_weight_map.get('stem_bn/gamma') or keras_weight_map.get('efficientnetb0/stem_bn/gamma')
        if python_gamma is not None:
            diff = np.abs(new_gamma.flatten() - python_gamma.astype(np.float32).flatten())
            print(f"\n  stem_bn/gamma max diff: {diff.max():.10f}")
            print(f"  stem_bn/gamma identical: {np.allclose(new_gamma, python_gamma, atol=1e-6)}")
        break
    offset += n * 4
