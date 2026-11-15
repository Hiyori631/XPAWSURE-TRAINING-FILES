import json
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    m = json.load(f)
layers = m['modelTopology']['config']['layers']
# Show ALL layers between block1a_activation and block1a_add (inclusive of boundaries)
found = False
for l in layers:
    n = l['name']
    if n == 'block1a_activation':
        found = True
    if found:
        inb = l.get('inbound_nodes', [[]])
        cfg = l.get('config', {})
        extra = ''
        if l['class_name'] == 'Conv2D':
            extra = f" filters={cfg.get('filters')} kernel={cfg.get('kernel_size')} strides={cfg.get('strides')} padding={cfg.get('padding')}"
        elif l['class_name'] == 'Activation':
            extra = f" activation={cfg.get('activation')}"
        elif l['class_name'] == 'Reshape':
            extra = f" target_shape={cfg.get('target_shape')}"
        elif l['class_name'] == 'DepthwiseConv2D':
            extra = f" kernel_size={cfg.get('kernel_size')} strides={cfg.get('strides')} padding={cfg.get('padding')}"
        elif l['class_name'] == 'BatchNormalization':
            extra = f" epsilon={cfg.get('epsilon')} momentum={cfg.get('momentum')}"
        elif l['class_name'] == 'Multiply':
            pass
        print(f"  {n:<35} {l['class_name']:<25} inbound={inb}{extra}")
    if n == 'block1a_add':
        break
