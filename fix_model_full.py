import json
import sys
import os
import numpy as np
import shutil
import tensorflow as tf

MODEL_PATH = "models/fine_tuned_best_effnet_clean_5class_s42.keras"
ORIGINAL_JSON = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model_original.json"
OUT_MODEL_JSON = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json"

KERAS3_STRIP_FIELDS = {'module', 'registered_name', 'build_config', 'compile_config', 'ragged', 'optional', 'synchronized', 'quantization_config'}
REGULARIZER_FIELDS = {'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint', 'beta_regularizer', 'gamma_regularizer', 'beta_constraint', 'gamma_constraint'}
BN_KERAS3_FIELDS = {'renorm', 'renorm_clipping', 'renorm_momentum'}
ALL_STRIP_FIELDS = KERAS3_STRIP_FIELDS | REGULARIZER_FIELDS | BN_KERAS3_FIELDS

def convert_initializer(init_cfg):
    if not isinstance(init_cfg, dict):
        return init_cfg
    cn = init_cfg.get('class_name', '')
    mapping = {
        'GlorotUniform': 'glorot_uniform', 'GlorotNormal': 'glorot_normal',
        'HeUniform': 'he_uniform', 'HeNormal': 'he_normal',
        'LecunUniform': 'lecun_uniform', 'LecunNormal': 'lecun_normal',
        'Orthogonal': 'orthogonal', 'RandomNormal': 'RandomNormal',
        'RandomUniform': 'RandomUniform', 'Zeros': 'zeros', 'Ones': 'ones',
        'Identity': 'identity',
    }
    if cn in mapping:
        return mapping[cn]
    if cn == 'VarianceScaling':
        cfg = init_cfg.get('config', {})
        scale = cfg.get('scale', 1.0)
        mode = cfg.get('mode', 'fan_in')
        dist = cfg.get('distribution', 'uniform')
        if scale == 2.0 and mode == 'fan_out' and dist == 'truncated_normal':
            return 'he_normal'
        if scale == 1.0 and mode == 'fan_in' and dist == 'uniform':
            return 'glorot_uniform'
        return 'glorot_uniform'
    return 'glorot_uniform'

def clean_config(cfg):
    if not isinstance(cfg, dict):
        return cfg
    cfg = {k: v for k, v in cfg.items() if k not in ALL_STRIP_FIELDS}
    for key in list(cfg.keys()):
        val = cfg[key]
        if isinstance(val, dict):
            if 'class_name' in val and 'module' in val:
                cfg[key] = convert_initializer(val)
            elif any(k in val for k in KERAS3_STRIP_FIELDS):
                cfg[key] = {k: v for k, v in val.items() if k not in KERAS3_STRIP_FIELDS}
    return cfg

def collect_leaf_layers_from_config(layers_list, prefix=""):
    result = []
    for ld in layers_list:
        name = ld.get('config', {}).get('name', ld.get('name', ''))
        full_path = f"{prefix}/{name}" if prefix else name
        if ld.get('class_name') == 'Functional':
            inner = ld.get('config', {}).get('layers', [])
            result.extend(collect_leaf_layers_from_config(inner, full_path))
        else:
            result.append((full_path, ld))
    return result

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"Total weights: {len(model.weights)}")

def collect_keras_layers(model, prefix=""):
    result = []
    for layer in model.layers:
        lp = f"{prefix}/{layer.name}" if prefix else layer.name
        if hasattr(layer, 'layers') and layer.layers:
            result.extend(collect_keras_layers(layer, lp))
        else:
            result.append((lp, layer))
    return result

keras_layers = collect_keras_layers(model)
print(f"Keras leaf layers: {len(keras_layers)}")

keras_map = {path: layer for path, layer in keras_layers}
keras_name_map = {}
for path, layer in keras_layers:
    keras_name_map[layer.name] = (path, layer)

def get_inbound_from_keras(layer, layer_path, keras_map):
    try:
        nodes = layer._inbound_nodes
    except AttributeError:
        return None
    if not nodes:
        return None
    result = []
    for node in nodes:
        node_inputs = []
        try:
            input_tensors = node.input_tensors
            items = input_tensors.items() if isinstance(input_tensors, dict) else enumerate(input_tensors)
            for _, tensor in items:
                src_name = None
                src_idx = 0
                tensor_idx = 0
                if hasattr(tensor, '_keras_history'):
                    kh = tensor._keras_history
                    src_name = kh[0].name
                    src_idx = kh[1] if len(kh) > 1 else 0
                    tensor_idx = kh[2] if len(kh) > 2 else 0
                elif hasattr(tensor, 'node') and tensor.node is not None:
                    node_ref = tensor.node
                    if hasattr(node_ref, 'layer'):
                        src_name = node_ref.layer.name
                if src_name and src_name in keras_name_map:
                    src_path, _ = keras_name_map[src_name]
                    node_inputs.append([src_path, src_idx, tensor_idx, {}])
                elif src_name:
                    node_inputs.append([src_name, src_idx, tensor_idx, {}])
        except Exception:
            return None
        if node_inputs:
            result.append(node_inputs)
    return result if result else None

print("Building inbound_nodes from Keras graph...")
inbound_map = {}
for path, layer in keras_layers:
    nodes = get_inbound_from_keras(layer, path, keras_map)
    if nodes:
        inbound_map[path] = nodes

print(f"Got inbound_nodes for {len(inbound_map)} layers")

print("Reading original model.json...")
with open(ORIGINAL_JSON, "r") as f:
    model_json = json.load(f)

topology = model_json["modelTopology"]
config = topology.get("config", {})

def strip_prefix_from_refs(inbound_nodes, prefix):
    if not inbound_nodes or not prefix:
        return
    for node in inbound_nodes:
        for ref in node:
            if isinstance(ref, list) and len(ref) >= 1 and isinstance(ref[0], str):
                if ref[0].startswith(prefix + '/'):
                    ref[0] = ref[0][len(prefix) + 1:]

def fix_inbound_nodes_format(layers_list):
    for ld in layers_list:
        if ld.get('class_name') == 'Functional':
            fix_inbound_nodes_format(ld.get('config', {}).get('layers', []))
        if 'inbound_nodes' in ld:
            for i, node in enumerate(ld['inbound_nodes']):
                ld['inbound_nodes'][i] = [ref[:3] for ref in node]

def fix_layers_recursive(layers_list, prefix=""):
    fixed = 0
    for ld in layers_list:
        name = ld.get('config', {}).get('name', ld.get('name', ''))
        full_path = f"{prefix}/{name}" if prefix else name

        for k in KERAS3_STRIP_FIELDS:
            ld.pop(k, None)

        if ld.get('class_name') == 'Functional':
            inner_cfg = ld.get('config', {})
            for k in KERAS3_STRIP_FIELDS:
                inner_cfg.pop(k, None)
            inner_layers = inner_cfg.get('layers', [])

            for key in ['input_layers', 'output_layers']:
                val = inner_cfg.get(key)
                if val and isinstance(val[0], str):
                    inner_cfg[key] = [[val[0], val[1] if len(val) > 1 else 0, val[2] if len(val) > 2 else 0]]

            fixed += fix_layers_recursive(inner_layers, full_path)
            for il in inner_layers:
                if il.get('inbound_nodes'):
                    strip_prefix_from_refs(il['inbound_nodes'], full_path)
        else:
            if 'config' in ld:
                ld['config'] = clean_config(ld['config'])

            if full_path in inbound_map:
                ld['inbound_nodes'] = inbound_map[full_path]
                fixed += 1
            elif name in inbound_map:
                ld['inbound_nodes'] = inbound_map[name]
                fixed += 1

    return fixed

topology['config'] = config
for key in ['input_layers', 'output_layers']:
    val = config.get(key)
    if val and isinstance(val[0], str):
        config[key] = [[val[0], val[1] if len(val) > 1 else 0, val[2] if len(val) > 2 else 0]]

layers_data = config.get("layers", [])
fixed = fix_layers_recursive(layers_data)
print(f"Fixed {fixed} inbound_nodes")

fix_inbound_nodes_format(layers_data)
print("Stripped {} from inbound_nodes format")

proper_names = []
for path, layer in keras_layers:
    for w in layer.weights:
        proper_names.append(f"{path}/{w.name}")

old_weights = model_json["weightsManifest"][0]["weights"]
if len(proper_names) == len(old_weights):
    for i, name in enumerate(proper_names):
        old_weights[i]["name"] = name
    model_json["weightsManifest"][0]["weights"] = old_weights
    print(f"Fixed {len(proper_names)} weight names")

with open(OUT_MODEL_JSON, "w") as f:
    json.dump(model_json, f)

print(f"Saved to {OUT_MODEL_JSON}")
print("Done!")
