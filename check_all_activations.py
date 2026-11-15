import json
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    m = json.load(f)
layers = m['modelTopology']['config']['layers']
for l in layers:
    if l['class_name'] in ('Conv2D', 'DepthwiseConv2D', 'Dense'):
        cfg = l.get('config', {})
        act = cfg.get('activation', 'linear')
        if act != 'linear':
            print(f"  {l['name']:<35} {l['class_name']:<25} activation={act}")
