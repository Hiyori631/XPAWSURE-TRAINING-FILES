"""
Compare each weight in the binary against prod_model.weights to find
exactly which weight(s) have wrong values.
"""
import tensorflow as tf
import numpy as np
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
MOBILE_MODEL_PATH = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json"
WEIGHTS_PATH = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\weights.bin"

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

# Load binary
with open(WEIGHTS_PATH, 'rb') as f:
    raw = f.read()

print(f"Binary size: {len(raw)} bytes")
print(f"prod_model weights: {len(prod_model.weights)}")
print(f"Manifest specs: {len(manifest_specs)}")

# Read binary sequentially using float32 (same as verify_mobile_model.py)
offset = 0
bin_weights = []
for spec in manifest_specs:
    shape = spec['shape']
    n_elements = 1
    for s in shape:
        n_elements *= s
    nbytes = n_elements * 4
    arr = np.frombuffer(raw[offset:offset+nbytes], dtype=np.float32).reshape(shape)
    bin_weights.append((spec['name'], arr))
    offset += nbytes

# Compare each weight
print(f"\n{'Index':>5} | {'Manifest Name':<40} | {'Prod Name':<30} | {'Shape OK':>7} | {'MaxDiff':>12} | {'Status'}")
print("-" * 120)

first_mismatch = True
for i, (bin_name, bin_arr) in enumerate(bin_weights):
    prod_arr = prod_model.weights[i].numpy().astype(np.float32)
    shape_ok = list(bin_arr.shape) == list(prod_arr.shape)
    
    if shape_ok:
        max_diff = np.max(np.abs(bin_arr - prod_arr))
        mad = np.mean(np.abs(bin_arr - prod_arr))
    else:
        max_diff = float('nan')
        mad = float('nan')
    
    status = "OK" if shape_ok and max_diff < 0.001 else "MISMATCH"
    
    if status == "MISMATCH" or (i < 20):
        prod_name = prod_model.weights[i].name.split('/')[-1] if '/' in prod_model.weights[i].name else prod_model.weights[i].name
        print(f"{i:5d} | {bin_name:<40} | {prod_name:<30} | {str(shape_ok):>7} | {max_diff:12.8f} | {status}")
    
    if status == "MISMATCH" and first_mismatch:
        first_mismatch = False
        print(f"\n  *** First mismatch at index {i} ***")
        print(f"  Manifest: {bin_name} shape={list(bin_arr.shape)}")
        print(f"  Prod:     {prod_model.weights[i].name} shape={list(prod_arr.shape)}")
        if shape_ok:
            print(f"  Binary first 10:    {bin_arr.flat[:10]}")
            print(f"  Prod_model first 10: {prod_arr.flat[:10]}")
            print(f"  Ratio first 10: {bin_arr.flat[:10] / (prod_arr.flat[:10] + 1e-30)}")

print(f"\nTotal bytes consumed: {offset}")
