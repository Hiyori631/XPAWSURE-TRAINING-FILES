"""
Verify mobile model format (model.json + weights.bin) against original Keras model.

Loads the TF.js-format model from mobile/assets/model/ and runs inference
independently using TensorFlow ops. Then compares with the original Keras
model output to determine if the model conversion is correct.

Usage:
    python verify_mobile_model.py [image_path]
"""

import tensorflow as tf
import numpy as np
import json
import struct
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === Configuration ===
MOBILE_MODEL_DIR = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model"
KERAS_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'class_names_clean_5class_s42.json')

# === Weight Loading ===

def load_weights_from_bin(weights_bin_path, weight_specs):
    """Load weights from a binary file according to weight specs."""
    with open(weights_bin_path, 'rb') as f:
        raw = f.read()

    offset = 0
    weights = {}
    for spec in weight_specs:
        name = spec['name']
        shape = spec['shape']
        dtype = spec.get('dtype', 'float32')

        n_elements = 1
        for s in shape:
            n_elements *= s

        nbytes = n_elements * 4  # float32 = 4 bytes
        raw_slice = raw[offset:offset + nbytes]
        if len(raw_slice) < nbytes:
            print(f"  WARNING: Weight '{name}' needs {nbytes} bytes but only {len(raw_slice)} available at offset {offset}")
            break

        arr = np.frombuffer(raw_slice, dtype=np.float32).reshape(shape)
        weights[name] = arr
        offset += nbytes

    print(f"Loaded {len(weights)} weights from binary ({offset} bytes)")
    return weights


def build_weight_map(weight_specs, raw_weights):
    """Build a map from layer_name -> {weight_name: numpy_array}."""
    layer_weights = {}
    for spec in weight_specs:
        full_name = spec['name']
        # Format: "layer_name/weight_name"
        parts = full_name.split('/', 1)
        if len(parts) == 2:
            layer_name, weight_name = parts
        else:
            layer_name = full_name
            weight_name = 'kernel'

        if layer_name not in layer_weights:
            layer_weights[layer_name] = {}
        layer_weights[layer_name][weight_name] = raw_weights[full_name]

    return layer_weights


# === Layer Implementations ===

def apply_rescaling(x, scale, offset):
    """Rescaling layer: x * scale + offset."""
    scale = tf.constant(scale, dtype=tf.float32)
    offset = tf.constant(offset, dtype=tf.float32)
    return x * scale + offset


def apply_normalization(x, mean, variance, epsilon=1e-7):
    """Normalization layer: (x - mean) / sqrt(variance + epsilon)."""
    mean = tf.constant(mean, dtype=tf.float32)
    variance = tf.constant(variance, dtype=tf.float32)
    return (x - mean) / tf.sqrt(variance + epsilon)


def apply_zero_padding(x, padding):
    """ZeroPadding2D layer."""
    # padding format: [[top, bottom], [left, right]]
    paddings = tf.constant([[0, 0], [padding[0][0], padding[0][1]],
                            [padding[1][0], padding[1][1]], [0, 0]])
    return tf.pad(x, paddings, mode='CONSTANT', constant_values=0)


def apply_conv2d(x, kernel, bias=None, strides=1, padding='valid', groups=1):
    """Conv2D layer."""
    if padding == 'same':
        # Use tf.nn.conv2d with padding='SAME'
        if groups > 1:
            # Depthwise-like conv with groups
            result = tf.nn.depthwise_conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME')
        else:
            result = tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME')
    else:
        if groups > 1:
            result = tf.nn.depthwise_conv2d(x, kernel, strides=[1, strides, strides, 1], padding='VALID')
        else:
            result = tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='VALID')

    if bias is not None:
        result = result + bias
    return result


def apply_depthwise_conv2d(x, kernel, strides=1, padding='valid'):
    """DepthwiseConv2D layer."""
    # Kernel shape: [H, W, C_in, depth_multiplier]
    if padding == 'same':
        result = tf.nn.depthwise_conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME')
    else:
        result = tf.nn.depthwise_conv2d(x, kernel, strides=[1, strides, strides, 1], padding='VALID')
    return result


def apply_batch_norm(x, gamma, beta, moving_mean, moving_variance, epsilon=0.001, momentum=0.99):
    """BatchNormalization layer in inference mode."""
    # Apply variance clamp like TF.js does
    moving_variance_clamped = tf.maximum(moving_variance, 0.0)
    return tf.nn.batch_normalization(x, moving_mean, moving_variance_clamped, beta, gamma, epsilon)


def apply_activation(x, activation_name):
    """Activation layer."""
    if activation_name == 'swish' or activation_name == 'silu':
        return x * tf.sigmoid(x)
    elif activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'sigmoid':
        return tf.sigmoid(x)
    elif activation_name == 'softmax':
        return tf.nn.softmax(x)
    elif activation_name == 'linear':
        return x
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def apply_global_avg_pool(x):
    """GlobalAveragePooling2D layer."""
    return tf.reduce_mean(x, axis=[1, 2])


def apply_reshape(x, target_shape):
    """Reshape layer."""
    # target_shape does not include batch dimension
    return tf.reshape(x, [x.shape[0]] + list(target_shape))


def apply_dense(x, kernel, bias=None):
    """Dense layer."""
    result = tf.matmul(x, kernel)
    if bias is not None:
        result = result + bias
    return result


# === Model Execution ===

def execute_mobile_model(layers_config, layer_weights, input_tensor):
    """
    Execute the mobile model layer by layer following the topology.
    
    Args:
        layers_config: List of layer configs from model.json
        layer_weights: Dict mapping layer_name -> {weight_name: numpy_array}
        input_tensor: Input tensor of shape [1, 224, 224, 3]
    
    Returns:
        Dict of layer_name -> output_tensor for checkpoint layers
    """
    outputs = {}  # layer_name -> tensor
    checkpoints = {}  # layer_name -> tensor (for comparison)

    # Checkpoint layers to record
    cp_set = {
        'rescaling', 'normalization', 'rescaling_1',
        'stem_conv', 'stem_bn', 'stem_activation',
        'block1a_dwconv', 'block1a_bn', 'block1a_activation',
        'block1a_se_reduce', 'block1a_se_expand', 'block1a_se_excite',
        'block1a_project_conv', 'block1a_project_bn',
        'block2a_expand_conv', 'block2a_expand_bn', 'block2a_expand_activation',
        'block2a_dwconv', 'block2a_bn', 'block2a_activation',
        'block2a_project_conv', 'block2a_project_bn',
        'block2b_expand_conv', 'block2b_expand_bn', 'block2b_expand_activation',
        'block2b_dwconv', 'block2b_bn', 'block2b_activation',
        'block2b_project_conv', 'block2b_project_bn', 'block2b_add',
        'block5c_add', 'block6d_add',
        'top_bn', 'top_activation',
        'global_average_pooling2d', 'batch_normalization',
        'dense', 'dense_1', 'dense_2', 'softmax',
    }

    for layer_cfg in layers_config:
        name = layer_cfg['name']
        cls = layer_cfg['class_name']
        cfg = layer_cfg['config']
        inb = layer_cfg.get('inbound_nodes', [[]])

        # Resolve inputs
        if not inb or not inb[0]:
            if cls == 'InputLayer':
                outputs[name] = input_tensor
                continue
            else:
                print(f"  WARNING: No inbound_nodes for {name} ({cls})")
                continue

        node = inb[0]  # First node
        input_tensors = []
        for ref in node:
            src_name = ref[0]
            if src_name not in outputs:
                print(f"  WARNING: Missing input '{src_name}' for {name}")
                break
            input_tensors.append(outputs[src_name])
        else:
            # All inputs resolved
            if len(input_tensors) == 1:
                x = input_tensors[0]
            else:
                x = input_tensors

            # Execute layer
            w = layer_weights.get(name, {})

            if cls == 'Rescaling':
                out = apply_rescaling(x, cfg['scale'], cfg.get('offset', 0))

            elif cls == 'Normalization':
                out = apply_normalization(x, w['mean'], w['variance'])

            elif cls == 'ZeroPadding2D':
                out = apply_zero_padding(x, cfg['padding'])

            elif cls == 'Conv2D':
                out = apply_conv2d(
                    x, w['kernel'], w.get('bias'),
                    strides=cfg['strides'][0] if isinstance(cfg['strides'], list) else cfg['strides'],
                    padding=cfg['padding'],
                    groups=cfg.get('groups', 1)
                )
                activation = cfg.get('activation')
                if activation and activation != 'linear':
                    out = apply_activation(out, activation)

            elif cls == 'DepthwiseConv2D':
                out = apply_depthwise_conv2d(
                    x, w['depthwise_kernel'],
                    strides=cfg['strides'][0] if isinstance(cfg['strides'], list) else cfg['strides'],
                    padding=cfg['padding']
                )

            elif cls == 'BatchNormalization':
                out = apply_batch_norm(
                    x, w['gamma'], w['beta'], w['moving_mean'], w['moving_variance'],
                    epsilon=cfg.get('epsilon', 0.001),
                    momentum=cfg.get('momentum', 0.99)
                )

            elif cls == 'Activation':
                out = apply_activation(x, cfg['activation'])

            elif cls == 'GlobalAveragePooling2D':
                out = apply_global_avg_pool(x)
                if cfg.get('keepdims', False):
                    out = tf.expand_dims(tf.expand_dims(out, 1), 1)

            elif cls == 'Reshape':
                out = apply_reshape(x, cfg['target_shape'])

            elif cls == 'Multiply':
                out = x[0] * x[1]

            elif cls == 'Add':
                out = x[0] + x[1]

            elif cls == 'Dropout':
                out = x  # No-op in inference

            elif cls == 'Dense':
                out = apply_dense(x, w['kernel'], w.get('bias'))
                activation = cfg.get('activation')
                if activation and activation != 'linear':
                    out = apply_activation(out, activation)

            elif cls == 'Softmax':
                out = tf.nn.softmax(x)

            else:
                print(f"  WARNING: Unknown layer class: {cls} ({name})")
                out = x

            outputs[name] = out

            # Record checkpoints
            if name in cp_set:
                checkpoints[name] = out

    return checkpoints


# === Comparison Functions ===

def compare_values(name, python_val, mobile_val):
    """Compare two numpy arrays and print statistics."""
    pv = np.array(python_val).flatten()
    mv = np.array(mobile_val).flatten()

    min_p, max_p, mean_p = pv.min(), pv.max(), pv.mean()
    min_m, max_m, mean_m = mv.min(), mv.max(), mv.mean()

    # Check for NaN
    has_nan_p = np.any(np.isnan(pv))
    has_nan_m = np.any(np.isnan(mv))

    # Compute ratio (avoiding division by zero)
    if abs(min_p) > 1e-10 and abs(min_m) > 1e-10:
        ratio_min = min_m / min_p if min_p != 0 else float('inf')
    else:
        ratio_min = float('inf')

    if abs(max_p) > 1e-10 and abs(max_m) > 1e-10:
        ratio_max = max_m / max_p if max_p != 0 else float('inf')
    else:
        ratio_max = float('inf')

    # Mean absolute difference
    if pv.shape == mv.shape:
        mad = np.mean(np.abs(pv - mv))
    else:
        mad = float('inf')

    # Max absolute difference
    if pv.shape == mv.shape:
        max_diff = np.max(np.abs(pv - mv))
    else:
        max_diff = float('inf')

    print(f"\n[{name}]")
    print(f"  Python:  min={min_p:12.6f}  max={max_p:12.6f}  mean={mean_p:12.6f}  NaN={'YES' if has_nan_p else 'no'}")
    print(f"  Mobile:  min={min_m:12.6f}  max={max_m:12.6f}  mean={mean_m:12.6f}  NaN={'YES' if has_nan_m else 'no'}")
    print(f"  Ratio:   min={ratio_min:12.4f}  max={ratio_max:12.4f}")
    print(f"  Diff:    MAD={mad:.8f}  MaxDiff={max_diff:.8f}")

    if mad < 0.001:
        print(f"  MATCH:   Values are very close (MAD < 0.001)")
    elif mad < 1.0:
        print(f"  CLOSE:   Values are somewhat close (MAD < 1.0)")
    else:
        print(f"  DIVERGED: Values are significantly different")


# === Main ===

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__),
        "ProcessedDataset224_Clean_5class", "Test", "Mange",
        "05d7a9f26dc88df3c7fcff811dc6da45.jpg"
    )

    print("=" * 60)
    print("MOBILE MODEL FORMAT VERIFICATION")
    print("=" * 60)

    # --- Step 1: Load mobile model.json ---
    mobile_model_path = os.path.join(MOBILE_MODEL_DIR, "model.json")
    print(f"\nLoading mobile model.json from: {mobile_model_path}")
    with open(mobile_model_path, 'r') as f:
        mobile_model_json = json.load(f)

    topology = mobile_model_json['modelTopology']
    layers_config = topology['config']['layers']
    weight_specs = mobile_model_json['weightsManifest'][0]['weights']

    print(f"  Layers: {len(layers_config)}")
    print(f"  Weight specs: {len(weight_specs)}")

    # --- Step 2: Load weights from weights.bin ---
    weights_bin_path = os.path.join(MOBILE_MODEL_DIR, "weights.bin")
    print(f"\nLoading weights.bin from: {weights_bin_path}")
    raw_weights = load_weights_from_bin(weights_bin_path, weight_specs)
    layer_weights = build_weight_map(weight_specs, raw_weights)
    print(f"  Layer weight groups: {len(layer_weights)}")

    # --- Step 3: Load test image ---
    print(f"\nLoading test image: {image_path}")
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_resized = tf.image.resize(img_array, (224, 224))
    input_tensor = tf.expand_dims(img_resized, 0)
    print(f"  Input: shape={input_tensor.shape}, min={tf.reduce_min(input_tensor).numpy():.4f}, max={tf.reduce_max(input_tensor).numpy():.4f}, mean={tf.reduce_mean(input_tensor).numpy():.4f}")

    # --- Step 4: Execute mobile model format ---
    print("\n" + "=" * 60)
    print("EXECUTING MOBILE MODEL FORMAT (layer by layer)")
    print("=" * 60)

    checkpoints = execute_mobile_model(layers_config, layer_weights, input_tensor)

    # --- Step 5: Load and execute original Keras model ---
    print("\n" + "=" * 60)
    print("EXECUTING ORIGINAL KERAS MODEL")
    print("=" * 60)

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

    keras_model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
    )

    # Get Python predictions
    python_output = keras_model(input_tensor, training=False)
    python_logits = python_output.numpy()[0]
    python_probs = tf.nn.softmax(python_logits).numpy()

    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    print(f"\nPython Keras output:")
    print(f"  Logits: {[f'{v:.6f}' for v in python_logits.tolist()]}")
    print(f"  Probs:  {[f'{v:.6f}' for v in python_probs.tolist()]}")
    pred = np.argmax(python_probs)
    print(f"  Predicted: {class_names[pred]} ({python_probs[pred]*100:.2f}%)")

    # --- Step 6: Get mobile model final output ---
    if 'softmax' in checkpoints:
        mobile_probs = checkpoints['softmax'].numpy().flatten()
        print(f"\nMobile model format output:")
        print(f"  Probs: {[f'{v:.6f}' for v in mobile_probs.tolist()]}")
        pred_m = np.argmax(mobile_probs)
        print(f"  Predicted: {class_names[pred_m]} ({mobile_probs[pred_m]*100:.2f}%)")
    else:
        print("\n  WARNING: softmax output not found in checkpoints!")
        # Try dense_2 output for logits
        if 'dense_2' in checkpoints:
            mobile_logits = checkpoints['dense_2'].numpy().flatten()
            mobile_probs = tf.nn.softmax(mobile_logits).numpy()
            print(f"  Dense_2 logits: {[f'{v:.6f}' for v in mobile_logits.tolist()]}")
            print(f"  Probs: {[f'{v:.6f}' for v in mobile_probs.tolist()]}")
            pred_m = np.argmax(mobile_probs)
            print(f"  Predicted: {class_names[pred_m]} ({mobile_probs[pred_m]*100:.2f}%)")

    # --- Step 7: Compare intermediate values ---
    print("\n" + "=" * 60)
    print("COMPARING INTERMEDIATE VALUES")
    print("=" * 60)

    # Build Python sub-models for the same checkpoints
    backbone = keras_model.get_layer('efficientnetb0')
    backbone_layer_map = {layer.name: layer for layer in backbone.layers}

    # Map mobile checkpoint names to Python layers
    python_checkpoints = {}
    python_checkpoint_map = {
        'rescaling': ('rescaling', 'backbone'),
        'normalization': ('normalization', 'backbone'),
        'rescaling_1': ('rescaling_1', 'backbone'),
        'stem_conv': ('stem_conv', 'backbone'),
        'stem_bn': ('stem_bn', 'backbone'),
        'stem_activation': ('stem_activation', 'backbone'),
        'block1a_dwconv': ('block1a_dwconv', 'backbone'),
        'block1a_bn': ('block1a_bn', 'backbone'),
        'block1a_activation': ('block1a_activation', 'backbone'),
        'block1a_se_reduce': ('block1a_se_reduce', 'backbone'),
        'block1a_se_expand': ('block1a_se_expand', 'backbone'),
        'block1a_se_excite': ('block1a_se_excite', 'backbone'),
        'block1a_project_conv': ('block1a_project_conv', 'backbone'),
        'block1a_project_bn': ('block1a_project_bn', 'backbone'),
        'block2a_expand_conv': ('block2a_expand_conv', 'backbone'),
        'block2a_expand_bn': ('block2a_expand_bn', 'backbone'),
        'block2a_expand_activation': ('block2a_expand_activation', 'backbone'),
        'block2a_dwconv': ('block2a_dwconv', 'backbone'),
        'block2a_bn': ('block2a_bn', 'backbone'),
        'block2a_activation': ('block2a_activation', 'backbone'),
        'block2a_project_conv': ('block2a_project_conv', 'backbone'),
        'block2a_project_bn': ('block2a_project_bn', 'backbone'),
        'block2b_expand_conv': ('block2b_expand_conv', 'backbone'),
        'block2b_expand_bn': ('block2b_expand_bn', 'backbone'),
        'block2b_expand_activation': ('block2b_expand_activation', 'backbone'),
        'block2b_dwconv': ('block2b_dwconv', 'backbone'),
        'block2b_bn': ('block2b_bn', 'backbone'),
        'block2b_activation': ('block2b_activation', 'backbone'),
        'block2b_project_conv': ('block2b_project_conv', 'backbone'),
        'block2b_project_bn': ('block2b_project_bn', 'backbone'),
        'block2b_add': ('block2b_add', 'backbone'),
        'block5c_add': ('block5c_add', 'backbone'),
        'block6d_add': ('block6d_add', 'backbone'),
        'top_bn': ('top_bn', 'backbone'),
        'top_activation': ('top_activation', 'backbone'),
        'global_average_pooling2d': ('global_average_pooling2d', 'head'),
        'batch_normalization': ('batch_normalization', 'head'),
        'dense': ('dense', 'head'),
        'dense_1': ('dense_1', 'head'),
        'dense_2': ('dense_2', 'head'),
        'softmax': ('softmax', 'head'),
    }

    print("\nComputing Python sub-model outputs...")
    for cp_name, (py_name, location) in python_checkpoint_map.items():
        try:
            if location == 'backbone':
                if py_name in backbone_layer_map:
                    layer = backbone_layer_map[py_name]
                    sub_model = tf.keras.Model(inputs=backbone.input, outputs=layer.output)
                    output = sub_model(input_tensor, training=False)
                    python_checkpoints[cp_name] = output
                else:
                    print(f"  SKIP: {py_name} not found in backbone")
            else:
                layer = keras_model.get_layer(py_name)
                sub_model = tf.keras.Model(inputs=keras_model.input, outputs=layer.output)
                output = sub_model(input_tensor, training=False)
                python_checkpoints[cp_name] = output
        except Exception as e:
            print(f"  ERROR getting Python output for {cp_name}: {e}")

    # Compare all matching checkpoints
    print("\n" + "-" * 60)
    print("CHECKPOINT COMPARISON (Mobile format vs Python Keras)")
    print("-" * 60)

    checkpoints_to_compare = [
        'rescaling', 'normalization', 'rescaling_1',
        'stem_conv', 'stem_bn', 'stem_activation',
        'block1a_dwconv', 'block1a_bn', 'block1a_activation',
        'block1a_se_reduce', 'block1a_se_expand', 'block1a_se_excite',
        'block1a_project_conv', 'block1a_project_bn',
        'block2a_expand_conv', 'block2a_expand_bn', 'block2a_expand_activation',
        'block2a_dwconv', 'block2a_bn', 'block2a_activation',
        'block2a_project_conv', 'block2a_project_bn',
        'block2b_expand_conv', 'block2b_expand_bn', 'block2b_expand_activation',
        'block2b_dwconv', 'block2b_bn', 'block2b_activation',
        'block2b_project_conv', 'block2b_project_bn', 'block2b_add',
        'block5c_add', 'block6d_add',
        'top_bn', 'top_activation',
        'global_average_pooling2d', 'batch_normalization',
        'dense', 'dense_1', 'dense_2',
    ]

    first_diverged = None
    for cp_name in checkpoints_to_compare:
        if cp_name in checkpoints and cp_name in python_checkpoints:
            mobile_val = checkpoints[cp_name].numpy()
            python_val = python_checkpoints[cp_name].numpy()
            compare_values(cp_name, python_val, mobile_val)

            # Check if diverged
            pv = python_val.flatten()
            mv = mobile_val.flatten()
            if pv.shape == mv.shape:
                mad = np.mean(np.abs(pv - mv))
                if mad > 1.0 and first_diverged is None:
                    first_diverged = cp_name
        elif cp_name in checkpoints:
            print(f"\n[{cp_name}] Python output not available for comparison")
        else:
            print(f"\n[{cp_name}] Mobile output not available")

    # --- Step 8: Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if first_diverged:
        print(f"\nFirst significantly diverged layer: {first_diverged}")
        print(f"This indicates the model conversion issue starts at this layer.")
    else:
        print("\nAll checkpoint values are close (MAD < 1.0)")
        print("The model format appears correct.")
        print("If predictions are still wrong, the issue may be in TF.js runtime behavior.")

    # Final prediction comparison
    if 'softmax' in checkpoints:
        mobile_probs = checkpoints['softmax'].numpy().flatten()
        print(f"\nPrediction comparison:")
        print(f"  Python:    {class_names[np.argmax(python_probs)]} ({python_probs[np.argmax(python_probs)]*100:.2f}%)")
        print(f"  Mobile fmt: {class_names[np.argmax(mobile_probs)]} ({mobile_probs[np.argmax(mobile_probs)]*100:.2f}%)")

        if np.argmax(python_probs) == np.argmax(mobile_probs):
            print(f"  SAME CLASS predicted!")
        else:
            print(f"  DIFFERENT CLASS predicted!")


if __name__ == '__main__':
    main()
