"""
Compare weight shapes between prod_model.weights and mobile manifest
to find where the order diverges.
"""
import tensorflow as tf
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
MOBILE_MODEL_PATH = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json"

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

inputs = model.input
outputs = tf.keras.layers.Softmax()(model.output)
prod_model = tf.keras.Model(inputs, outputs, name="xpawsure_model")

with open(MOBILE_MODEL_PATH, 'r') as f:
    mobile_json = json.load(f)

manifest_specs = mobile_json['weightsManifest'][0]['weights']

print(f"prod_model.weights count: {len(prod_model.weights)}")
print(f"manifest specs count: {len(manifest_specs)}")

mismatches = 0
for i, (w, spec) in enumerate(zip(prod_model.weights, manifest_specs)):
    prod_shape = list(w.numpy().shape)
    mob_shape = spec['shape']
    prod_dtype = str(w.numpy().dtype)
    mob_dtype = spec['dtype']
    
    if prod_shape != mob_shape or prod_dtype != mob_dtype:
        print(f"MISMATCH at index {i}:")
        print(f"  prod: name={w.name} shape={prod_shape} dtype={prod_dtype}")
        print(f"  mob:  name={spec['name']} shape={mob_shape} dtype={mob_dtype}")
        mismatches += 1

print(f"\nTotal mismatches: {mismatches}")

# Also check: what weight is at index 13 in prod_model?
print(f"\nprod_model weight at index 13: {prod_model.weights[13].name} shape={prod_model.weights[13].shape}")
print(f"manifest spec at index 13: {manifest_specs[13]['name']} shape={manifest_specs[13]['shape']}")

# Check the backbone weights specifically
backbone = model.get_layer('efficientnetb0')
print(f"\nbackbone weights count: {len(backbone.weights)}")

# Check the first few backbone weights
for i, w in enumerate(backbone.weights[:5]):
    print(f"  backbone weight {i}: {w.name} shape={list(w.numpy().shape)}")

# Check: does the backbone have a 'count' variable?
for w in backbone.weights:
    if 'count' in w.name:
        print(f"\n  FOUND count in backbone: {w.name} shape={list(w.numpy().shape)} dtype={w.numpy().dtype}")
