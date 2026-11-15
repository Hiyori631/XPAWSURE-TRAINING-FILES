import json
import sys
import os


def flatten_model(input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    with open(input_path, 'r') as f:
        d = json.load(f)

    topo = d['modelTopology']
    config = topo['config']
    top_layers = config['layers']

    func_layer = None
    func_idx = None
    for i, l in enumerate(top_layers):
        if l['class_name'] == 'Functional':
            func_layer = l
            func_idx = i
            break

    if func_layer is None:
        print("No Functional layer found, nothing to flatten")
        return

    func_cfg = func_layer['config']
    internal_layers = func_cfg['layers']
    func_input_name = func_cfg['input_layers'][0]
    func_output_name = func_cfg['output_layers'][0]

    flat_layers = []

    top_input_layer = top_layers[0]
    flat_layers.append(top_input_layer)
    top_input_name = top_input_layer['config']['name']

    for il in internal_layers:
        layer_name = il['config']['name']
        if layer_name == func_input_name:
            continue

        new_layer = json.loads(json.dumps(il))

        inb = new_layer.get('inbound_nodes', [])
        if isinstance(inb, list) and inb:
            for i, node in enumerate(inb):
                if isinstance(node, list) and node and isinstance(node[0], list):
                    for j, conn in enumerate(node):
                        if isinstance(conn, list) and len(conn) >= 1:
                            if conn[0] == func_input_name:
                                conn[0] = top_input_name
                elif isinstance(node, dict):
                    for arg in node.get('args', []):
                        cfg = arg.get('config', {}) if isinstance(arg, dict) else {}
                        kh = cfg.get('keras_history', [])
                        if kh and kh[0] == func_input_name:
                            kh[0] = top_input_name

        flat_layers.append(new_layer)

    for outer_layer in top_layers[1:]:
        if outer_layer['class_name'] == 'Functional':
            continue

        new_layer = json.loads(json.dumps(outer_layer))

        inb = new_layer.get('inbound_nodes', [])
        if isinstance(inb, list) and inb:
            for i, node in enumerate(inb):
                if isinstance(node, list) and node and isinstance(node[0], list):
                    for j, conn in enumerate(node):
                        if isinstance(conn, list) and len(conn) >= 1:
                            if conn[0] == func_layer['config']['name']:
                                conn[0] = func_output_name
                elif isinstance(node, dict):
                    for arg in node.get('args', []):
                        cfg = arg.get('config', {}) if isinstance(arg, dict) else {}
                        kh = cfg.get('keras_history', [])
                        if kh and kh[0] == func_layer['config']['name']:
                            kh[0] = func_output_name

        flat_layers.append(new_layer)

    config['layers'] = flat_layers

    with open(output_path, 'w') as f:
        json.dump(d, f)

    print("Flattened %s -> %s" % (input_path, output_path))
    print("  Before: %d top-level (1 Functional with %d internal)" % (len(top_layers), len(internal_layers)))
    print("  After: %d top-level layers" % len(flat_layers))

    names = [l['config']['name'] for l in flat_layers]
    assert len(names) == len(set(names)), "DUPLICATE LAYER NAMES FOUND!"
    print("  All layer names unique: OK")
    print("  First layer: %s" % flat_layers[0]['config']['name'])
    print("  Last layer: %s" % flat_layers[-1]['config']['name'])


if __name__ == '__main__':
    paths = sys.argv[1:] or [
        os.path.join('mobile', 'assets', 'model', 'model.json'),
        os.path.join('models', 'tfjs_model', 'model.json'),
    ]
    for p in paths:
        if os.path.exists(p):
            flatten_model(p)
        else:
            print("Skipping %s (not found)" % p)
