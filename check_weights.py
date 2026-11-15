import tensorflow as tf
import numpy as np
import json, os
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

model = tf.keras.models.load_model(
    "models/fine_tuned_best_effnet_clean_5class_s42.keras",
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)

backbone = model.get_layer("efficientnetb0")
stem_conv = backbone.get_layer("stem_conv")
kernel_np = stem_conv.weights[0].numpy()
print("Python kernel shape:", kernel_np.shape)
print("Python kernel first 10 values:", kernel_np.flatten()[:10])

with open(r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\model.json") as f:
    mobile_json = json.load(f)

with open(r"C:\Users\Lenovo LOQ\xPawSure-Capstone\mobile\assets\model\weights.bin", "rb") as f:
    raw = f.read()

offset = 0
for spec in mobile_json["weightsManifest"][0]["weights"]:
    n = 1
    for s in spec["shape"]:
        n *= s
    if spec["name"] == "stem_conv/kernel":
        mobile_kernel = np.frombuffer(raw[offset:offset+n*4], dtype=np.float32).reshape(spec["shape"])
        print("Mobile kernel shape:", mobile_kernel.shape)
        print("Mobile kernel first 10 values:", mobile_kernel.flatten()[:10])
        diff = np.abs(kernel_np.flatten() - mobile_kernel.flatten())
        print("Max diff:", diff.max())
        print("Mean diff:", diff.mean())
        print("Identical:", np.allclose(kernel_np, mobile_kernel))
        break
    offset += n * 4

# Also check: stem_bn weights
print("\n--- stem_bn ---")
stem_bn = backbone.get_layer("stem_bn")
for i, w in enumerate(stem_bn.weights):
    print(f"Python stem_bn weight {i} ({w.name}): shape={w.shape} first5={w.numpy().flatten()[:5]}")

offset = 0
for spec in mobile_json["weightsManifest"][0]["weights"]:
    n = 1
    for s in spec["shape"]:
        n *= s
    if spec["name"].startswith("stem_bn/"):
        mobile_w = np.frombuffer(raw[offset:offset+n*4], dtype=np.float32).reshape(spec["shape"])
        print(f"Mobile {spec['name']}: shape={spec['shape']} first5={mobile_w.flatten()[:5]}")
    offset += n * 4

# Check all Conv2D kernel shapes
print("\n--- ALL Conv2D kernel shapes ---")
for spec in mobile_json["weightsManifest"][0]["weights"]:
    if "kernel" in spec["name"] and "depthwise" not in spec["name"] and "se_" not in spec["name"]:
        print(f"  {spec['name']}: {spec['shape']}")
