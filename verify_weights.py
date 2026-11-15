"""Verify weight binary alignment between manifest and actual binary data.
Loads the mobile model.json manifest, reads weights.bin, and for each weight
in order, prints its name, shape, and first few values. Then compares with
the Keras model's actual weight values."""
import json
import struct
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load mobile manifest
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    mobile = json.load(f)

weights_manifest = mobile['weightsManifest'][0]
weight_specs = weights_manifest['weights']
weights_path = r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\weights.bin'

print(f"Manifest entries: {len(weight_specs)}")
print(f"weights.bin size: {os.path.getsize(weights_path)} bytes")

# Read binary data
with open(weights_path, 'rb') as f:
    all_bytes = f.read()

print(f"Binary data loaded: {len(all_bytes)} bytes")

# Parse each weight from binary data
DTYPE_MAP = {
    'float32': ('f', 4),
    'float16': ('e', 2),
    'int32': ('i', 4),
    'int8': ('b', 1),
    'bool': ('?', 1),
}

offset = 0
manifest_weights = []
for i, spec in enumerate(weight_specs):
    name = spec['name']
    shape = spec['shape']
    dtype = spec.get('dtype', 'float32')
    
    if dtype not in DTYPE_MAP:
        print(f"  WARNING: unknown dtype '{dtype}' for {name}")
        manifest_weights.append(None)
        continue
    
    fmt, byte_size = DTYPE_MAP[dtype]
    num_elements = 1
    for s in shape:
        num_elements *= s
    total_bytes = num_elements * byte_size
    
    if offset + total_bytes > len(all_bytes):
        print(f"  ERROR: {name} needs {total_bytes} bytes but only {len(all_bytes) - offset} remaining at offset {offset}")
        break
    
    # Parse values
    raw = all_bytes[offset:offset + total_bytes]
    values = struct.unpack(f'<{num_elements}{fmt}', raw)
    
    manifest_weights.append({
        'name': name,
        'shape': shape,
        'dtype': dtype,
        'offset': offset,
        'size': total_bytes,
        'values': values,
        'min': min(values) if values else 0,
        'max': max(values) if values else 0,
        'mean': sum(values) / len(values) if values else 0,
    })
    
    offset += total_bytes

print(f"\nParsed {len(manifest_weights)} weights from binary, total {offset} bytes")
if offset < len(all_bytes):
    print(f"WARNING: {len(all_bytes) - offset} extra bytes at end of binary file!")

# Now load Keras model and get its weights
print("\n" + "="*60)
print("LOADING KERAS MODEL FOR COMPARISON")
print("="*60)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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
    r'models/fine_tuned_best_effnet_clean_5class_s42.keras',
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

# Get all weight values from Keras model, organized by layer
keras_weights = {}
for layer in model.layers:
    # Check nested layers (inside efficientnetb0)
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            for w in sublayer.weights:
                keras_weights[w.name] = w.numpy()
    for w in layer.weights:
        keras_weights[w.name] = w.numpy()

print(f"Keras weight variables: {len(keras_weights)}")

# Build mapping: mobile manifest name -> Keras variable name
# The mobile manifest has names like "stem_conv/kernel", "stem_bn/gamma"
# The Keras variables have names like "block1a_dwconv/depthwise_kernel:0"
# We need to map between them

# First, let's see what Keras variable names look like
print("\nFirst 10 Keras variable names:")
for i, name in enumerate(list(keras_weights.keys())[:10]):
    print(f"  {name}")

# Build a mapping from manifest name to Keras variable
# Strategy: strip the ":0" suffix and match by layer name + weight name
def manifest_to_keras_name(manifest_name):
    """Convert manifest name like 'stem_bn/gamma' to Keras name like 'stem_bn/gamma:0'"""
    return manifest_name + ':0'

# Compare first 20 weights
print("\n" + "="*60)
print("COMPARING FIRST 20 WEIGHTS (manifest vs Keras)")
print("="*60)

mismatches = 0
checked = 0
for i, mw in enumerate(manifest_weights[:30]):
    if mw is None:
        continue
    
    manifest_name = mw['name']
    keras_name = manifest_to_keras_name(manifest_name)
    
    if keras_name not in keras_weights:
        # Try without :0
        keras_name2 = manifest_name
        if keras_name2 not in keras_weights:
            print(f"\n  {manifest_name}: NOT FOUND in Keras model")
            # Try to find by partial match
            for kn in keras_weights:
                if manifest_name.split('/')[-1] in kn and manifest_name.split('/')[0] in kn:
                    print(f"    Possible match: {kn}")
                    break
            continue
    
    keras_val = keras_weights[keras_name]
    manifest_val = list(mw['values'])
    
    # Compare shapes
    keras_shape = list(keras_val.shape)
    manifest_shape = mw['shape']
    
    if keras_shape != manifest_shape:
        print(f"\n  {manifest_name}: SHAPE MISMATCH! manifest={manifest_shape} keras={keras_shape}")
        mismatches += 1
        continue
    
    # Compare values
    keras_flat = keras_val.flatten().tolist()[:10]
    manifest_flat = manifest_val[:10]
    
    # Check if values match (within float32 tolerance)
    match = True
    for j in range(min(len(keras_flat), len(manifest_flat))):
        if abs(keras_flat[j] - manifest_flat[j]) > 1e-5:
            match = False
            break
    
    status = "MATCH" if match else "MISMATCH"
    if not match:
        mismatches += 1
    
    checked += 1
    print(f"\n  {manifest_name}: {status}")
    print(f"    manifest: shape={manifest_shape}, first5={[f'{v:.6f}' for v in manifest_flat[:5]]}")
    print(f"    keras:    shape={keras_shape}, first5={[f'{v:.6f}' for v in keras_flat[:5]]}")

print(f"\n\nSummary: checked={checked}, mismatches={mismatches}")

# Now do a more thorough comparison - check ALL weights
print("\n" + "="*60)
print("FULL WEIGHT COMPARISON (all 330 weights)")
print("="*60)

total_mismatches = 0
total_checked = 0
total_shape_mismatches = 0
not_found = 0

for mw in manifest_weights:
    if mw is None:
        continue
    
    manifest_name = mw['name']
    keras_name = manifest_to_keras_name(manifest_name)
    
    if keras_name not in keras_weights:
        not_found += 1
        continue
    
    keras_val = keras_weights[keras_name]
    manifest_val = mw['values']
    
    keras_shape = list(keras_val.shape)
    manifest_shape = mw['shape']
    
    if keras_shape != manifest_shape:
        total_shape_mismatches += 1
        continue
    
    # Compare first value as a quick check
    keras_first = float(keras_val.flatten()[0])
    manifest_first = manifest_val[0]
    
    total_checked += 1
    if abs(keras_first - manifest_first) > 1e-4:
        total_mismatches += 1
        if total_mismatches <= 10:
            print(f"  VALUE MISMATCH: {manifest_name}")
            print(f"    manifest first={manifest_first:.6f}, keras first={keras_first:.6f}")

print(f"\nFull comparison: checked={total_checked}, value_mismatches={total_mismatches}, shape_mismatches={total_shape_mismatches}, not_found={not_found}")
