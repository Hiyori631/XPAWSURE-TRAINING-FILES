import json
d = json.load(open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json'))
content = json.dumps(d)
checks = [
    ('batch_shape', 'batch_shape' in content),
    ('DTypePolicy', 'DTypePolicy' in content),
    ('L2 reg', '"L2"' in content),
    ('L1 reg', '"L1"' in content),
    ('ragged', '"ragged"' in content),
    ('optional', '"optional"' in content),
    ('synchronized', '"synchronized"' in content),
    ('quantization_config', 'quantization_config' in content),
    ('Functional layer', '"Functional"' in content),
    ('keras_history', 'keras_history' in content),
]
for name, present in checks:
    print('%s: %s' % (name, 'PRESENT' if present else 'clean'))
print('Size: %d bytes' % len(content))
