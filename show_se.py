import json
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    m = json.load(f)
layers = m['modelTopology']['config']['layers']
for l in layers:
    n = l['name']
    if 'se' in n or n in ('block1a_activation', 'block1a_dwconv'):
        inb = l.get('inbound_nodes', [[]])
        print(f"  {n:<35} class={l['class_name']:<25} inbound={inb}")
