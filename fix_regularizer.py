import json, re, os

for p in [
    r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json',
    r'C:\Users\Lenovo LOQ\Downloads\XPAWSURE - Copy - Copy\models\tfjs_model\model.json',
]:
    if not os.path.exists(p):
        continue
    with open(p) as f:
        d = json.load(f)
    content = json.dumps(d)
    l2 = content.count('"L2"')
    l1 = content.count('"L1"')
    print('%s: L2=%d, L1=%d' % (os.path.basename(p), l2, l1))

    if l2 > 0 or l1 > 0:
        for layer in d['modelTopology']['config']['layers']:
            cfg = layer.get('config', {})
            for key in ['kernel_regularizer', 'bias_regularizer']:
                reg = cfg.get(key)
                if reg and isinstance(reg, dict):
                    cn = reg.get('class_name', '')
                    if cn in ('L2', 'L1'):
                        print('  Found %s regularizer in layer %s (class %s)' % (cn, cfg.get('name'), layer['class_name']))
                        print('  Config: %s' % json.dumps(reg))

        for layer in d['modelTopology']['config']['layers']:
            cfg = layer.get('config', {})
            for key in ['kernel_regularizer', 'bias_regularizer']:
                reg = cfg.get(key)
                if reg and isinstance(reg, dict):
                    cn = reg.get('class_name', '')
                    if cn == 'L2':
                        reg['class_name'] = 'l2'
                    elif cn == 'L1':
                        reg['class_name'] = 'l1'

        with open(p, 'w') as f:
            json.dump(d, f)
        print('  Fixed: L2->l2, L1->l1')
