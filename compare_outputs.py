import tensorflow as tf
import numpy as np
import json
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_weights = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}
alpha = tf.constant([class_weights[i] for i in range(5)], dtype=tf.float32)

def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        p_t = tf.gather_nd(probs, tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1))
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if alpha is not None:
            alpha_t = tf.gather(alpha, tf.cast(y_true, tf.int32))
            focal = alpha_t * focal
        return tf.reduce_mean(focal)
    return loss_fn

print("Loading Keras model...")
model = tf.keras.models.load_model(
    "models/fine_tuned_best_effnet_clean_5class_s42.keras",
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

with open("models/class_names_clean_5class_s42.json") as f:
    class_names = json.load(f)
print(f"Classes: {class_names}")

# Access the EfficientNetB0 backbone (nested Functional model)
backbone = model.get_layer('efficientnetb0')
print(f"\nBackbone type: {type(backbone).__name__}")
print(f"Backbone layers: {len(backbone.layers)}")

# List all backbone layers with their class names
print("\n=== ALL BACKBONE LAYERS ===")
for i, layer in enumerate(backbone.layers):
    cls = layer.__class__.__name__
    has_weights = len(layer.weights) > 0
    w_info = ""
    if has_weights:
        w_info = f" [{len(layer.weights)} weights]"
    print(f"  {i:3d}: {layer.name:40s} {cls:25s}{w_info}")

image_path = sys.argv[1] if len(sys.argv) > 1 else "ProcessedDataset224_Clean_5class\\Test\\Mange\\05d7a9f26dc88df3c7fcff811dc6da45.jpg"
print(f"\nImage: {image_path}")

from PIL import Image
img = Image.open(image_path).convert("RGB")
img_array = np.array(img, dtype=np.float32)
img_resized = tf.image.resize(img_array, (224, 224))
input_tensor = tf.expand_dims(img_resized, 0)

print(f"\nInput: shape={input_tensor.shape}, dtype={input_tensor.dtype}, min={tf.reduce_min(input_tensor).numpy():.4f}, max={tf.reduce_max(input_tensor).numpy():.4f}, mean={tf.reduce_mean(input_tensor).numpy():.4f}")

# Build a full model from backbone input to each checkpoint layer
# Key: these layers are INSIDE the backbone, so we use backbone.input and layer.output

print("\n" + "="*60)
print("LAYER-BY-LAYER OUTPUT TRACE (inside EfficientNetB0 backbone)")
print("="*60)

checkpoint_names = [
    'rescaling',
    'normalization',
    'rescaling_1',
    'stem_conv_pad',
    'stem_conv',
    'stem_bn',
    'stem_activation',
    'block1a_dwconv',
    'block1a_bn',
    'block1a_activation',
    'block1a_se_reduce',
    'block1a_se_expand',
    'block1a_se_excite',
    'block1a_project_conv',
    'block1a_project_bn',
    'block1a_proj_drop',
    'block1a_add',
    'block2b_expand_conv',
    'block2b_expand_bn',
    'block2b_expand_activation',
    'block2b_dwconv',
    'block2b_bn',
    'block2b_activation',
    'block2b_se_reduce',
    'block2b_se_expand',
    'block2b_se_excite',
    'block2b_project_conv',
    'block2b_project_bn',
    'block2b_proj_drop',
    'block2b_add',
    'block5c_expand_conv',
    'block5c_expand_bn',
    'block5c_expand_activation',
    'block5c_dwconv',
    'block5c_bn',
    'block5c_activation',
    'block5c_se_reduce',
    'block5c_se_expand',
    'block5c_se_excite',
    'block5c_project_conv',
    'block5c_project_bn',
    'block5c_proj_drop',
    'block5c_add',
    'block6d_expand_conv',
    'block6d_expand_bn',
    'block6d_expand_activation',
    'block6d_dwconv',
    'block6d_bn',
    'block6d_activation',
    'block6d_se_reduce',
    'block6d_se_expand',
    'block6d_se_excite',
    'block6d_project_conv',
    'block6d_project_bn',
    'block6d_proj_drop',
    'block6d_add',
    'top_conv',
    'top_bn',
    'top_activation',
]

# Build layer name -> layer object lookup for the backbone
backbone_layer_map = {}
for layer in backbone.layers:
    backbone_layer_map[layer.name] = layer

for cp_name in checkpoint_names:
    if cp_name not in backbone_layer_map:
        print(f"\n[{cp_name}] -> NOT FOUND in backbone")
        continue
    try:
        layer = backbone_layer_map[cp_name]
        # Build a sub-model from backbone.input to this layer's output
        sub_model = tf.keras.Model(inputs=backbone.input, outputs=layer.output)
        output = sub_model(input_tensor, training=False)
        out_np = output.numpy().flatten()
        has_nan = np.any(np.isnan(out_np))
        print(f"\n[{cp_name}] layer={cp_name} ({layer.__class__.__name__})")
        print(f"  shape={output.shape}, min={np.nanmin(out_np):.6f}, max={np.nanmax(out_np):.6f}, mean={np.nanmean(out_np):.6f}, NaN={'YES' if has_nan else 'no'}")
        if out_np.shape[0] <= 20:
            print(f"  values={out_np.tolist()}")
        else:
            print(f"  first10={out_np[:10].tolist()}")
            print(f"  last5={out_np[-5:].tolist()}")
    except Exception as e:
        print(f"\n[{cp_name}] -> ERROR: {e}")

# Also print classification head outputs
print("\n" + "="*60)
print("CLASSIFICATION HEAD")
print("="*60)

head_checkpoints = [
    'global_average_pooling2d',
    'batch_normalization',
    'dropout',
    'dense',
    'batch_normalization_1',
    'dropout_1',
    'dense_1',
    'batch_normalization_2',
    'dropout_2',
    'dense_2',
]

for cp_name in head_checkpoints:
    try:
        layer = model.get_layer(cp_name)
        sub_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        output = sub_model(input_tensor, training=False)
        out_np = output.numpy().flatten()
        has_nan = np.any(np.isnan(out_np))
        print(f"\n[{cp_name}] layer={cp_name} ({layer.__class__.__name__})")
        print(f"  shape={output.shape}, min={np.nanmin(out_np):.6f}, max={np.nanmax(out_np):.6f}, mean={np.nanmean(out_np):.6f}, NaN={'YES' if has_nan else 'no'}")
        if out_np.shape[0] <= 20:
            print(f"  values={out_np.tolist()}")
        else:
            print(f"  first10={out_np[:10].tolist()}")
            print(f"  last5={out_np[-5:].tolist()}")
    except Exception as e:
        print(f"\n[{cp_name}] -> ERROR: {e}")

# Final prediction
print("\n" + "="*60)
print("FINAL OUTPUT")
print("="*60)

full_output = model(input_tensor, training=False)
logits = full_output.numpy()[0]
probs = tf.nn.softmax(logits).numpy()
print(f"Logits: {[f'{v:.6f}' for v in logits.tolist()]}")
print(f"Probs:  {[f'{v:.6f}' for v in probs.tolist()]}")
pred = np.argmax(probs)
print(f"Predicted: {class_names[pred]} ({probs[pred]*100:.2f}%)")
