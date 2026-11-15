import numpy as np
import json
import struct

weights_bin = r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\weights.bin"
with open(weights_bin, "rb") as f:
    raw = f.read()

print(f"Total binary size: {len(raw)} bytes")

with open(r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json") as f:
    mobile_json = json.load(f)

specs = mobile_json["weightsManifest"][0]["weights"]

# Read each weight spec and check what the data looks like
offset = 0
for i, spec in enumerate(specs):
    name = spec["name"]
    shape = spec["shape"]
    n_elements = 1
    for s in shape:
        n_elements *= s
    nbytes = n_elements * 4

    data = raw[offset:offset + nbytes]
    if len(data) < nbytes:
        print(f"  {i}: {name} ({shape}) - TRUNCATED, need {nbytes} bytes, {len(data)} available")
        break

    arr = np.frombuffer(data, dtype=np.float32)
    print(f"  {i}: {name:45s} shape={str(shape):20s} nbytes={nbytes:6d}  offset={offset:6d}  first3={arr[:3].tolist()}")

    offset += nbytes

# Now specifically check: read the FIRST 40 bytes of the binary file as floats
print("\nFirst 40 bytes of weights.bin as float32:")
raw_floats = np.frombuffer(raw[:40], dtype=np.float32)
for i, v in enumerate(raw_floats):
    print(f"  byte offset {i*4:2d}: {v}")

# Check normalization/count specifically
# normalization/mean: [3] = 12 bytes (offset 0-11)
# normalization/variance: [3] = 12 bytes (offset 12-23)
# normalization/count: [] = 4 bytes (offset 24-27) - but is it really 4?
count_bytes = raw[24:32]
print(f"\nnormalization/count raw bytes (offset 24-31): {count_bytes.hex()}")
count_as_float = np.frombuffer(count_bytes[:4], dtype=np.float32)
print(f"normalization/count as float32 (first 4 bytes): {count_as_float}")
count_as_int32 = np.frombuffer(count_bytes[:4], dtype=np.int32)
print(f"normalization/count as int32 (first 4 bytes): {count_as_int32}")
count_as_float64 = np.frombuffer(count_bytes[:8], dtype=np.float64)
print(f"normalization/count as float64 (8 bytes): {count_as_float64}")
count_as_int64 = np.frombuffer(count_bytes[:8], dtype=np.int64)
print(f"normalization/count as int64 (8 bytes): {count_as_int64}")

# Also check the original weights.bin
orig_weights_bin = r"C:\Users\Lenovo LOQ\Downloads\XPAWSURE - Copy - Copy\models\tfjs_model\weights.bin"
with open(orig_weights_bin, "rb") as f:
    orig_raw = f.read()

print(f"\nOriginal binary size: {len(orig_raw)} bytes")
print(f"Mobile binary size:   {len(raw)} bytes")
print(f"Binary files identical: {orig_raw == raw}")

# Check original first 40 bytes
print("\nOriginal first 40 bytes as float32:")
orig_floats = np.frombuffer(orig_raw[:40], dtype=np.float32)
for i, v in enumerate(orig_floats):
    print(f"  byte offset {i*4:2d}: {v}")

# Compare original and mobile at offset 24
print(f"\nOriginal bytes at offset 24-31: {orig_raw[24:32].hex()}")
print(f"Mobile bytes at offset 24-31:   {raw[24:32].hex()}")
