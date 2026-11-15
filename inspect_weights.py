"""Inspect the TF.js model.json weight manifest vs the Keras model layer weights.
Compare the original (from convert_to_tfjs) with the mobile (flattened) manifest."""
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load mobile model.json
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    mobile = json.load(f)

# Load original model.json
with open(r'models\tfjs_model\model.json') as f:
    original = json.load(f)

mobile_weights = mobile['weightsManifest'][0]['weights']
original_weights = original['weightsManifest'][0]['weights']

print(f"Mobile weights: {len(mobile_weights)}")
print(f"Original weights: {len(original_weights)}")

# Check if mobile weight names are unique
mobile_names = [w['name'] for w in mobile_weights]
print(f"\nMobile unique names: {len(set(mobile_names))}/{len(mobile_names)}")
if len(set(mobile_names)) != len(mobile_names):
    from collections import Counter
    dupes = [(n, c) for n, c in Counter(mobile_names).items() if c > 1]
    print(f"  DUPLICATES: {dupes[:10]}")

# Check if original weight names are unique
orig_names = [w['name'] for w in original_weights]
print(f"Original unique names: {len(set(orig_names))}/{len(orig_names)}")
if len(set(orig_names)) != len(orig_names):
    from collections import Counter
    dupes = [(n, c) for n, c in Counter(orig_names).items() if c > 1]
    print(f"  DUPLICATES (first 10): {dupes[:10]}")

# Now simulate TF.js loadWeights shortening logic
print("\n=== TF.js loadWeights simulation (mobile manifest) ===")
# Build nameToWeight map from mobile manifest
mobile_map = {}
for w in mobile_weights:
    mobile_map[w['name']] = w
print(f"Mobile map size: {len(mobile_map)} (all unique = good)")

print("\n=== TF.js loadWeights simulation (original manifest) ===")
# Build nameToWeight map from original manifest
orig_map = {}
for w in original_weights:
    orig_map[w['name']] = w  # Last one wins if duplicates
print(f"Original map size: {len(orig_map)} (was {len(original_weights)} entries)")

# Check how many unique names in original
from collections import Counter
orig_counts = Counter(orig_names)
print(f"Original name frequency distribution:")
for name, count in orig_counts.most_common(20):
    print(f"  '{name}': {count} occurrences")

# Now simulate what happens when TF.js loads mobile model weights
# It iterates model.layers -> layer.weights -> gets variable names like "layer_name/weight_name"
# Then tries to match against the manifest map
# Let's load the Keras model and get the actual variable names
print("\n=== Keras model layer structure ===")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(
    r'models/fine_tuned_best_effnet_clean_5class_s42.keras',
    compile=False
)

# Get all weight variable names from the Keras model
keras_var_names = []
for layer in model.layers:
    for w in layer.weights:
        keras_var_names.append(w.name)
    # Also check nested layers
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            for w in sublayer.weights:
                keras_var_names.append(w.name)

print(f"Keras total weight variables: {len(keras_var_names)}")
print(f"First 10: {keras_var_names[:10]}")
print(f"Last 10: {keras_var_names[-10:]}")

# Now let's see what TF.js would use as variable names for each layer
# In TF.js, weight variable name = "{layer_name}/{weight_name}"
# where weight_name comes from addWeight() calls
# Let's check what the mobile model.json topology expects
mobile_topology = mobile['modelTopology']
if 'config' in mobile_topology:
    config = mobile_topology['config']
    if 'layers' in config:
        layer_names = [l['config']['name'] for l in config['layers']]
        print(f"\nMobile model topology layer count: {len(layer_names)}")
        print(f"First 5: {layer_names[:5]}")
        print(f"Last 5: {layer_names[-5:]}")

# Check BatchNormalization layers and their config
print("\n=== BatchNormalization layer configs ===")
bn_layers = [l for l in config['layers'] if l['class_name'] == 'BatchNormalization']
for bn in bn_layers[:5]:
    cfg = bn['config']
    print(f"  {cfg['name']}: gamma={cfg.get('gamma', 'N/A')}, beta={cfg.get('beta', 'N/A')}, moving_mean={cfg.get('moving_mean', 'N/A')}, moving_variance={cfg.get('moving_variance', 'N/A')}")

# Check what weight names TF.js would generate for each BN layer
# TF.js BatchNormalization.build() creates: gamma, beta, moving_mean, moving_variance
# So variable names would be: "{layer_name}/gamma", "{layer_name}/beta", etc.
# Check if these match the mobile manifest
print("\n=== Weight name matching (mobile manifest) ===")
matched = 0
unmatched = 0
for bn in bn_layers:
    name = bn['config']['name']
    for suffix in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
        tfjs_name = f"{name}/{suffix}"
        if tfjs_name in mobile_map:
            matched += 1
        else:
            unmatched += 1
            print(f"  UNMATCHED: {tfjs_name}")

# Check DepthwiseConv2D layers
print("\n=== DepthwiseConv2D weight names ===")
dw_layers = [l for l in config['layers'] if l['class_name'] == 'DepthwiseConv2D']
for dw in dw_layers[:3]:
    name = dw['config']['name']
    # TF.js DepthwiseConv2D creates: depthwise_kernel (not kernel)
    for suffix in ['depthwise_kernel']:
        tfjs_name = f"{name}/{suffix}"
        in_mobile = tfjs_name in mobile_map
        print(f"  {tfjs_name}: {'MATCH' if in_mobile else 'NO MATCH'}")

# Check Conv2D layers
print("\n=== Conv2D weight names ===")
conv_layers = [l for l in config['layers'] if l['class_name'] == 'Conv2D']
for conv in conv_layers[:3]:
    name = conv['config']['name']
    for suffix in ['kernel', 'bias']:
        tfjs_name = f"{name}/{suffix}"
        in_mobile = tfjs_name in mobile_map
        print(f"  {tfjs_name}: {'MATCH' if in_mobile else 'NO MATCH'}")
