"""Compare mobile manifest weight values against original tfjs manifest.
Both use the same weights.bin file, so the raw bytes should match.
If the binary offset order differs, weights will be wrong."""
import json
import struct
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load mobile manifest
with open(r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json') as f:
    mobile = json.load(f)

# Load original manifest
with open(r'models\tfjs_model\model.json') as f:
    original = json.load(f)

mobile_specs = mobile['weightsManifest'][0]['weights']
original_specs = original['weightsManifest'][0]['weights']

mobile_bin = r'C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\weights.bin'
original_bin = r'models\tfjs_model\weights.bin'

mobile_data = open(mobile_bin, 'rb').read()
original_data = open(original_bin, 'rb').read()

print(f"Mobile binary: {len(mobile_data)} bytes")
print(f"Original binary: {len(original_data)} bytes")
print(f"Binaries identical: {mobile_data == original_data}")

DTYPE_MAP = {'float32': ('f', 4), 'float16': ('e', 2), 'int32': ('i', 4)}

def parse_weights(data, specs):
    weights = []
    offset = 0
    for spec in specs:
        name = spec['name']
        shape = spec['shape']
        dtype = spec.get('dtype', 'float32')
        fmt, byte_size = DTYPE_MAP[dtype]
        num_elements = 1
        for s in shape:
            num_elements *= s
        total_bytes = num_elements * byte_size
        
        raw = data[offset:offset + total_bytes]
        values = struct.unpack(f'<{num_elements}{fmt}', raw)
        
        weights.append({
            'name': name,
            'shape': shape,
            'offset': offset,
            'size': total_bytes,
            'first5': list(values[:5]),
            'min': min(values),
            'max': max(values),
        })
        offset += total_bytes
    return weights

mobile_weights = parse_weights(mobile_data, mobile_specs)
original_weights = parse_weights(original_data, original_specs)

print(f"\nMobile weights parsed: {len(mobile_weights)}")
print(f"Original weights parsed: {len(original_weights)}")

# The key test: do the first N bytes of each binary match?
# If the binaries are identical, then the weight order is the same,
# and the issue is NOT binary alignment.

# But let's also verify by comparing specific weights.
# The mobile manifest has unique names; the original has duplicates.
# The first weight in mobile is "normalization/mean" with shape [3].
# The first weight in original is "mean" with shape [3].
# These should have the same bytes.

print("\n" + "="*60)
print("FIRST 10 WEIGHTS COMPARISON (mobile vs original)")
print("="*60)

for i in range(min(10, len(mobile_weights), len(original_weights))):
    mw = mobile_weights[i]
    ow = original_weights[i]
    
    match = mw['first5'] == ow['first5']
    status = "MATCH" if match else "MISMATCH"
    
    print(f"\n  [{i}] {mw['name']} vs {ow['name']}: {status}")
    print(f"    Mobile:  shape={mw['shape']}, offset={mw['offset']}, first5={mw['first5']}")
    print(f"    Original: shape={ow['shape']}, offset={ow['offset']}, first5={ow['first5']}")

# Now check ALL weights
print("\n" + "="*60)
print("FULL COMPARISON (all 330 weights)")
print("="*60)

mismatches = 0
for i in range(min(len(mobile_weights), len(original_weights))):
    mw = mobile_weights[i]
    ow = original_weights[i]
    
    if mw['first5'] != ow['first5']:
        mismatches += 1
        if mismatches <= 15:
            print(f"\n  MISMATCH at [{i}]: {mw['name']} vs {ow['name']}")
            print(f"    Mobile first5={mw['first5']}")
            print(f"    Original first5={ow['first5']}")

print(f"\nTotal mismatches: {mismatches}/{min(len(mobile_weights), len(original_weights))}")

# Also compare binary content at specific offsets
print("\n" + "="*60)
print("BINARY CONTENT AT KEY OFFSETS")
print("="*60)

# Find stem_bn weights in both manifests
for mw in mobile_weights:
    if mw['name'] == 'stem_bn/moving_mean':
        print(f"\nMobile stem_bn/moving_mean: offset={mw['offset']}, first5={mw['first5']}")
        # Read raw bytes
        raw = mobile_data[mw['offset']:mw['offset']+mw['size']]
        print(f"  Raw bytes (first 20): {list(raw[:20])}")
        break

for ow in original_weights:
    if ow['name'] == 'moving_mean':
        # This is the first moving_mean in the original (which one?)
        print(f"\nOriginal moving_mean: offset={ow['offset']}, first5={ow['first5']}")
        raw = original_data[ow['offset']:ow['offset']+ow['size']]
        print(f"  Raw bytes (first 20): {list(raw[:20])}")
        break

# Check: are the raw bytes at mobile stem_bn/moving_mean the same as original moving_mean?
print("\n" + "="*60)
print("BINARY CONTENT AT OFFSET 0 (first weight)")
print("="*60)
print(f"Mobile first 40 bytes: {list(mobile_data[:40])}")
print(f"Original first 40 bytes: {list(original_data[:40])}")
print(f"First 40 bytes match: {mobile_data[:40] == original_data[:40]}")
