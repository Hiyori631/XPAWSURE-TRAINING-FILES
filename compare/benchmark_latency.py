"""Fair CPU inference-latency benchmark for all comparison checkpoints.
Warmup runs excluded. Updates compare/results/*_metrics.json in place.
"""
import os, json, glob, time
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RES = os.path.join(ROOT, "compare", "results")
DATASET = "ProcessedDataset224_Clean_5class"

custom_weights = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}
alpha = tf.constant([custom_weights[i] for i in range(5)], dtype=tf.float32)


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


# gather test images once
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET}/Test", image_size=(224, 224), batch_size=1, shuffle=False,
)
images = np.concatenate([x.numpy() for x, _ in test_ds], axis=0)
print(f"Loaded {len(images)} test images")

for p in sorted(glob.glob(os.path.join(RES, "*_metrics.json"))):
    m = json.load(open(p))
    ckpt = m.get("checkpoint")
    if not ckpt or not os.path.exists(ckpt):
        print(f"SKIP {p}: checkpoint missing ({ckpt})")
        continue
    model = tf.keras.models.load_model(
        ckpt, custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}, compile=False,
    )
    # warmup
    for i in range(5):
        model.predict(images[i:i+1], verbose=0)
    lat = []
    idx = np.linspace(0, len(images) - 1, 50).astype(int)
    for i in idx:
        t = time.perf_counter()
        model.predict(images[i:i+1], verbose=0)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    m["inference_ms_mean"] = round(float(lat.mean()), 2)
    m["inference_ms_std"] = round(float(lat.std()), 2)
    m["inference_ms_p50"] = round(float(np.percentile(lat, 50)), 2)
    m["inference_protocol"] = "CPU, batch=1, 50 images, 5 warmup runs excluded"
    json.dump(m, open(p, "w"), indent=2)
    print(f"{m['label']:16s} s{m.get('seed')}: {m['inference_ms_mean']:.1f} ± {m['inference_ms_std']:.1f} ms "
          f"(p50 {m['inference_ms_p50']:.1f})")
    del model

print("Done.")
