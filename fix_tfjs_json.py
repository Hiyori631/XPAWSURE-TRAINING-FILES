import json
import sys
import os

KERAS3_ONLY_FIELDS = {'ragged', 'optional', 'synchronized', 'quantization_config'}


def fix_inbound_nodes(layer):
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
                if kh and len(kh) == 3:
                    connections.append([kh[0], kh[1], kh[2], kwargs])
                elif kh and len(kh) == 2:
                    connections.append([kh[0], kh[1], 0, kwargs])
            layer['inbound_nodes'][i] = connections if connections else [[]]


def fix_config_fields(cfg):
    for key in list(cfg.keys()):
        if key in KERAS3_ONLY_FIELDS:
            del cfg[key]
    dtype = cfg.get('dtype')
    if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
        cfg['dtype'] = dtype.get('config', {}).get('name', 'float32')
    for reg_key in ('kernel_regularizer', 'bias_regularizer', 'activity_regularizer'):
        reg = cfg.get(reg_key)
        if isinstance(reg, dict):
            cn = reg.get('class_name', '')
            if cn == 'L2':
                reg['class_name'] = 'l2'
            elif cn == 'L1':
                reg['class_name'] = 'l1'
            for rk in list(reg.keys()):
                if rk in ('module', 'registered_name'):
                    del reg[rk]


def fix_layers_recursive(topology_config):
    layers = topology_config.get('layers', [])
    for layer in layers:
        fix_inbound_nodes(layer)
        fix_config_fields(layer.get('config', {}))

        if layer.get('class_name') == 'Functional':
            nested_config = layer.get('config', {})
            fix_layers_recursive(nested_config)


def fix_model_json(input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    with open(input_path, 'r') as f:
        d = json.load(f)

    topology = d.get('modelTopology', {})
    config = topology.get('config', {})

    fix_layers_recursive(config)

    layers = config.get('layers', [])
    for layer in layers:
        cfg = layer.get('config', {})
        if 'batch_shape' in cfg and 'batch_input_shape' not in cfg:
            cfg['batch_input_shape'] = cfg.pop('batch_shape')

    with open(output_path, 'w') as f:
        json.dump(d, f)

    fixed_count = sum(1 for l in layers
                      for node in l.get('inbound_nodes', [])
                      if isinstance(node, list))
    print("Fixed %s -> %s (%d layers, %d inbound_nodes fixed)" % (
        input_path, output_path, len(layers), fixed_count))
    return True


if __name__ == '__main__':
    paths = sys.argv[1:] or [
        os.path.join('mobile', 'assets', 'model', 'model.json'),
        os.path.join('models', 'tfjs_model', 'model.json'),
    ]
    for p in paths:
        if os.path.exists(p):
            fix_model_json(p)
        else:
            print("Skipping %s (not found)" % p)
