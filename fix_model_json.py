import json
import sys
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/fine_tuned_best_effnet_clean_5class_s42.keras"
OLD_MODEL_JSON = os.path.join("C:\\Users\\Lenovo LOQ\\xPawSure-Capstone\\mobile\\assets\\model", "model.json")
OUT_MODEL_JSON = os.path.join("C:\\Users\\Lenovo LOQ\\xPawSure-Capstone\\mobile\\assets\\model", "model.json")
BACKUP_JSON = os.path.join("C:\\Users\\Lenovo LOQ\\xPawSure-Capstone\\mobile\\assets\\model", "model_original.json")

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print(f"Total weights: {len(model.weights)}")

def collect_weight_names(model, prefix=""):
    """Recursively collect weight names from all layers."""
    names = []
    for layer in model.layers:
        layer_path = f"{prefix}/{layer.name}" if prefix else layer.name
        if hasattr(layer, 'layers') and layer.layers:
            # Nested model/functional - recurse
            names.extend(collect_weight_names(layer, layer_path))
        else:
            for w in layer.weights:
                names.append(f"{layer_path}/{w.name}")
    return names

proper_names = collect_weight_names(model)
print(f"Built {len(proper_names)} proper names from layer hierarchy")

print("\nFirst 20 proper names:")
for i, n in enumerate(proper_names[:20]):
    print(f"  [{i}] {n}")

print("\nLast 10 proper names:")
for i, n in enumerate(proper_names[-10:], len(proper_names)-10):
    print(f"  [{i}] {n}")

print(f"\nReading existing model.json...")
with open(OLD_MODEL_JSON, "r") as f:
    model_json = json.load(f)

import shutil
shutil.copy2(OLD_MODEL_JSON, BACKUP_JSON)
print(f"Backed up original to {BACKUP_JSON}")

old_weights = model_json["weightsManifest"][0]["weights"]
print(f"Old manifest has {len(old_weights)} entries")

if len(proper_names) != len(old_weights):
    print(f"ERROR: Weight count mismatch! Proper={len(proper_names)}, Manifest={len(old_weights)}")
    sys.exit(1)

for i, name in enumerate(proper_names):
    old_weights[i]["name"] = name

model_json["weightsManifest"][0]["weights"] = old_weights

with open(OUT_MODEL_JSON, "w") as f:
    json.dump(model_json, f)

print(f"\nSaved fixed model.json to {OUT_MODEL_JSON}")
print("Done!")
