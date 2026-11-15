import tensorflow as tf
import numpy as np
import json
import os

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
    "models/fine_tuned_best_effnet_clean_5class_s42.keras",
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

inputs = model.input
outputs = tf.keras.layers.Softmax()(model.output)
prod_model = tf.keras.Model(inputs, outputs, name="xpawsure_model")

out_dir = "models/tfjs_model"
os.makedirs(out_dir, exist_ok=True)

model_json = prod_model.to_json()
config = json.loads(model_json)

KERAS3_ONLY_FIELDS = {'ragged', 'optional', 'synchronized', 'quantization_config'}

def fix_keras3_inbound(layer):
    inb = layer.get('inbound_nodes')
    if not isinstance(inb, list):
        return
    for i, node in enumerate(inb):
        if isinstance(node, dict):
            args = node.get('args', [])
            kwargs = node.get('kwargs', {})
            connections = []
            for arg in args:
                cfg = arg.get('config', {}) if isinstance(arg, dict) else {}
                kh = cfg.get('keras_history', [])
                if kh and len(kh) >= 3:
                    connections.append([kh[0], kh[1], kh[2], kwargs])
                elif kh and len(kh) == 2:
                    connections.append([kh[0], kh[1], 0, kwargs])
            layer['inbound_nodes'][i] = connections if connections else [[]]

def fix_keras3_topology(top_config):
    for layer in top_config.get('layers', []):
        fix_keras3_inbound(layer)
        cfg = layer.get('config', {})
        for key in list(cfg.keys()):
            if key in KERAS3_ONLY_FIELDS:
                del cfg[key]
        dtype = cfg.get('dtype')
        if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
            cfg['dtype'] = dtype.get('config', {}).get('name', 'float32')
        if layer.get('class_name') == 'Functional':
            fix_keras3_topology(layer['config'])

fix_keras3_topology(config)
for layer in config.get('layers', []):
    cfg = layer.get('config', {})
    if 'batch_shape' in cfg and 'batch_input_shape' not in cfg:
        cfg['batch_input_shape'] = cfg.pop('batch_shape')

def flatten_functional(config):
    top_layers = config.get('layers', [])
    func_layer = None
    func_idx = None
    for i, l in enumerate(top_layers):
        if l.get('class_name') == 'Functional':
            func_layer = l
            func_idx = i
            break
    if func_layer is None:
        return
    func_cfg = func_layer['config']
    internal_layers = func_cfg['layers']
    func_input_name = func_cfg['input_layers'][0]
    func_output_name = func_cfg['output_layers'][0]
    top_input_name = top_layers[0]['config']['name']
    flat = [top_layers[0]]
    for il in internal_layers:
        if il['config']['name'] == func_input_name:
            continue
        new_layer = json.loads(json.dumps(il))
        for node in new_layer.get('inbound_nodes', []):
            if isinstance(node, list) and node and isinstance(node[0], list):
                for conn in node:
                    if isinstance(conn, list) and conn[0] == func_input_name:
                        conn[0] = top_input_name
        flat.append(new_layer)
    for outer in top_layers[1:]:
        if outer['class_name'] == 'Functional':
            continue
        new_layer = json.loads(json.dumps(outer))
        for node in new_layer.get('inbound_nodes', []):
            if isinstance(node, list) and node and isinstance(node[0], list):
                for conn in node:
                    if isinstance(conn, list) and conn[0] == func_layer['config']['name']:
                        conn[0] = func_output_name
        flat.append(new_layer)
    config['layers'] = flat
    print("Flattened to %d top-level layers" % len(flat))

flatten_functional(config)

model_json_tfjs = {
    "format": "layers-model",
    "generatedBy": "keras v" + tf.__version__,
    "convertedBy": None,
    "modelTopology": config,
    "weightsManifest": []
}

weight_values = []
weight_names = []
for w in prod_model.weights:
    weight_values.append(w.numpy())
    weight_names.append(w.name)

with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
    for val in weight_values:
        f.write(val.tobytes())

manifest_entries = []
for name, val in zip(weight_names, weight_values):
    manifest_entries.append({
        "name": name,
        "shape": list(val.shape),
        "dtype": "float32",
    })

model_json_tfjs["weightsManifest"] = [{
    "paths": ["weights.bin"],
    "weights": manifest_entries,
}]

with open(os.path.join(out_dir, "model.json"), "w") as f:
    json.dump(model_json_tfjs, f)

size_json = os.path.getsize(os.path.join(out_dir, "model.json"))
size_bin = os.path.getsize(os.path.join(out_dir, "weights.bin"))
print(f"Model saved to {out_dir}/")
print(f"  model.json: {size_json} bytes")
print(f"  weights.bin: {size_bin} bytes ({size_bin / 1024 / 1024:.1f} MB)")

labels = ["Allergic Dermatitis", "Bacterial", "Fungal", "Hotspot", "Mange"]
with open(os.path.join(out_dir, "labels.txt"), "w") as f:
    for label in labels:
        f.write(label + "\n")
print(f"  labels.txt: {labels}")

print("\nVerifying model outputs...")
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
pred = prod_model.predict(test_input, verbose=0)
print(f"  Output shape: {pred.shape}")
print(f"  Output sum: {pred.sum():.6f} (should be 1.0)")
print(f"  Output: {pred}")

print("\nDone!")
