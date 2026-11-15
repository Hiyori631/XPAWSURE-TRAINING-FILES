import json
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    m = json.load(f)
layers = m['modelTopology']['config']['layers']
for l in layers:
    n = l['name']
    if n in ('block1a_se_reduce', 'block1a_se_expand', 'block2a_se_reduce', 'block2a_se_expand', 'block7a_se_reduce', 'block7a_se_expand'):
        cfg = l.get('config', {})
        print(f"  {n}: activation={cfg.get('activation')}, filters={cfg.get('filters')}")
