"""Backbone-agnostic training for the XPawSure model comparison study.

Replicates the production EfficientNetB0 recipe (train_effnet.py /
fine_tune_effnet.py) with a swappable backbone. Writes ONLY under compare/.
"""
import os, sys, json, argparse, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, required=True,
                    choices=["effnet", "mobilenet", "resnet50", "densenet"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stage", type=str, required=True, choices=["frozen", "finetune"])
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max_batches", type=int, default=None,
                    help="Limit batches per dataset (smoke test only)")
parser.add_argument("--dataset", type=str, default="ProcessedDataset224_Clean_5class",
                    help="Dataset root with Train/Valid/Test subdirs")
parser.add_argument("--tag_prefix", type=str, default="",
                    help="Prefix for output tags, e.g. 'local_' for ablation runs")
args = parser.parse_args()

BACKBONE = args.backbone
SEED = args.seed
STAGE = args.stage

BACKBONES = {
    "effnet": dict(
        builder=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input,
        layer_name="efficientnetb0",
        label="EfficientNetB0",
    ),
    "mobilenet": dict(
        builder=tf.keras.applications.MobileNetV2,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
        layer_name="mobilenetv2_1.00_224",
        label="MobileNetV2",
    ),
    "resnet50": dict(
        builder=tf.keras.applications.ResNet50,
        preprocess=tf.keras.applications.resnet50.preprocess_input,
        layer_name="resnet50",
        label="ResNet50",
    ),
    "densenet": dict(
        builder=tf.keras.applications.DenseNet121,
        preprocess=tf.keras.applications.densenet.preprocess_input,
        layer_name="densenet121",
        label="DenseNet121",
    ),
}
CFG = BACKBONES[BACKBONE]

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
import random as _random
_random.seed(SEED)

DATASET_PATH = args.dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CKPT_DIR = os.path.join(ROOT, "compare", "ckpt")
RES_DIR = os.path.join(ROOT, "compare", "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

TAG = f"{args.tag_prefix}{BACKBONE}_s{SEED}"
FROZEN_CKPT = os.path.join(CKPT_DIR, f"{TAG}_frozen_best.keras")
BEST_CKPT = os.path.join(CKPT_DIR, f"{TAG}_best.keras")
CONFIG_JSON = os.path.join(RES_DIR, f"{TAG}_config.json")
METRICS_JSON = os.path.join(RES_DIR, f"{TAG}_metrics.json")

if STAGE == "frozen":
    EPOCHS = args.epochs if args.epochs else 100
else:
    EPOCHS = args.epochs if args.epochs else 80


def load_config():
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(CONFIG_JSON, "w") as f:
        json.dump(cfg, f, indent=2)


print("=" * 60)
print(f"BACKBONE: {CFG['label']} | SEED: {SEED} | STAGE: {STAGE} | EPOCHS: {EPOCHS}")
print("=" * 60)
print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED,
)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Valid", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False,
)

print(f"Classes: {class_names}")
assert NUM_CLASSES in (4, 5), f"Expected 4 or 5 classes, got {NUM_CLASSES}"

if NUM_CLASSES == 5:
    custom_weights = {0: 3.0, 1: 2.0, 2: 2.5, 3: 1.0, 4: 1.3}  # AD, Bact, Fung, Hot, Mange
else:
    custom_weights = {0: 2.0, 1: 2.5, 2: 1.0, 3: 1.3}  # Bact, Fung, Hot, Mange (alphabetical)
alpha_tensor = tf.constant([custom_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

if args.max_batches:
    train_ds = train_ds.take(args.max_batches)
    valid_ds = valid_ds.take(args.max_batches)
    test_ds = test_ds.take(args.max_batches)


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


class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, start_ratio=0.7):
        super().__init__()
        self.start_ratio = start_ratio
        self.swa_weights = None
        self.swa_count = 0
        self.start_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            total = self.params.get("epochs", EPOCHS)
            self.start_epoch = int(total * self.start_ratio)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            if self.swa_weights is None:
                self.swa_weights = [tf.identity(w).numpy() for w in self.model.trainable_weights]
            else:
                for i, w in enumerate(self.model.trainable_weights):
                    self.swa_weights[i] += w.numpy()
            self.swa_count += 1

    def on_train_end(self, logs=None):
        if self.swa_count > 1:
            for i, w in enumerate(self.model.trainable_weights):
                w.assign(self.swa_weights[i] / self.swa_count)
            print(f"\nSWA applied: averaged {self.swa_count} checkpoints from epoch {self.start_epoch}")


# ---------------- augmentation (identical to production recipe) ----------------
if STAGE == "frozen":
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ])
    SWA_RATIO = 0.7
else:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.05),
    ])
    SWA_RATIO = 0.6


def augment(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_augmentation(images, training=True)
    images = tf.clip_by_value(images, 0, 255)
    images = images / 255.0
    images = tf.image.random_hue(images, max_delta=0.05)
    images = tf.image.random_saturation(images, lower=0.7, upper=1.3)
    images = images * 255.0
    return images, labels


train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

cfg = load_config()
cfg.update({
    "backbone": BACKBONE,
    "label": CFG["label"],
    "seed": SEED,
    "dataset": DATASET_PATH,
    "image_size": 224,
    "batch_size": BATCH_SIZE,
    "class_names": class_names,
    "class_weights": {str(k): v for k, v in custom_weights.items()},
    "loss": "focal_loss gamma=2.0 with class alpha",
    "augmentation": "production recipe (frozen: rot0.15/zoom0.1/contrast0.15/brightness0.1/translate0.05; finetune: rot0.1/zoom0.05/contrast0.1/brightness0.05; both: hue0.05, sat0.7-1.3)",
})

# ---------------- build / load model ----------------
if STAGE == "frozen":
    print(f"\nBuilding {CFG['label']} (frozen backbone, ImageNet weights)...")
    base_model = CFG["builder"](input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    cfg["base_layer_name"] = base_model.name
    cfg["backbone_total_layers"] = len(base_model.layers)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = CFG["preprocess"](inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    model = models.Model(inputs, outputs)

    total_steps = EPOCHS * max(len(train_ds), 1)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-5
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
        metrics=["accuracy"],
    )
    ckpt_path = FROZEN_CKPT
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
        CSVLogger(os.path.join(RES_DIR, f"{TAG}_frozen_log.csv")),
        SWACallback(start_ratio=SWA_RATIO),
    ]
else:
    print(f"\nLoading frozen checkpoint: {FROZEN_CKPT}")
    if not os.path.exists(FROZEN_CKPT):
        sys.exit(f"ERROR: frozen checkpoint not found: {FROZEN_CKPT}. Run stage=frozen first.")
    model = tf.keras.models.load_model(
        FROZEN_CKPT, custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha_tensor.numpy())}
    )
    base_layer_name = cfg.get("base_layer_name", CFG["layer_name"])
    base_model = model.get_layer(base_layer_name)

    total_layers = len(base_model.layers)
    unfreeze_n = max(1, int(round(0.25 * total_layers)))
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_n:]:
        layer.trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_layers = [l.name for l in base_model.layers if l.trainable]
    print(f"Unfrozen {len(trainable_layers)}/{total_layers} backbone layers (last {unfreeze_n}, BN frozen)")
    cfg["unfreeze_policy"] = f"last {unfreeze_n} of {total_layers} (~25%), BN frozen"
    cfg["unfrozen_layer_count"] = len(trainable_layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
        metrics=["accuracy"],
    )
    ckpt_path = BEST_CKPT
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True),
        CSVLogger(os.path.join(RES_DIR, f"{TAG}_finetune_log.csv")),
    ]

model.summary()
save_config(cfg)

print("\n" + "=" * 60)
print(f"Starting {CFG['label']} {STAGE} training...")
print("=" * 60)

t0 = time.time()
history = model.fit(
    train_ds_aug, validation_data=valid_ds, epochs=EPOCHS,
    callbacks=callbacks, verbose=1,
)
duration = time.time() - t0
print(f"\n{STAGE} training duration: {duration:.1f}s ({duration/60:.1f} min)")

# reload best checkpoint
model = tf.keras.models.load_model(
    ckpt_path, custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha_tensor.numpy())}
)

cfg = load_config()
cfg[f"duration_s_{STAGE}"] = round(duration, 1)
cfg[f"epochs_{STAGE}"] = EPOCHS
cfg[f"best_val_accuracy_{STAGE}"] = float(max(history.history.get("val_accuracy", [0.0])))
save_config(cfg)

# curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Valid")
ax1.set_title(f"{CFG['label']} {STAGE} - Accuracy")
ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Valid")
ax2.set_title(f"{CFG['label']} {STAGE} - Loss")
ax2.set_xlabel("Epoch"); ax2.legend()
plt.savefig(os.path.join(RES_DIR, f"{TAG}_{STAGE}_curves.png"))
plt.close()

# ---------------- evaluation (test set: ONLY at finetune stage) ----------------
if STAGE == "finetune":
    print("\nEvaluating on Test Set (final evaluation)...")
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    labels_idx = list(range(NUM_CLASSES))
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    # inference latency (per image, CPU)
    lat = []
    single_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=1, shuffle=False,
    ).take(50)
    for images, _ in single_ds:
        t = time.time()
        model.predict(images, verbose=0)
        lat.append((time.time() - t) * 1000)
    lat = np.array(lat)

    metrics = {
        "backbone": BACKBONE,
        "label": CFG["label"],
        "seed": SEED,
        "stage": "finetune",
        "dataset": DATASET_PATH,
        "class_names": class_names,
        "test_accuracy": round(test_acc, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_precision": round(report["weighted avg"]["precision"], 4),
        "weighted_recall": round(report["weighted avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            } for c in class_names
        },
        "confusion_matrix": cm.tolist(),
        "params_total": int(model.count_params()),
        "model_size_bytes": os.path.getsize(ckpt_path),
        "duration_s_frozen": cfg.get("duration_s_frozen"),
        "duration_s_finetune": cfg.get("duration_s_finetune"),
        "inference_ms_mean": round(float(lat.mean()), 3),
        "inference_ms_std": round(float(lat.std()), 3),
        "unfreeze_policy": cfg.get("unfreeze_policy"),
        "checkpoint": os.path.relpath(ckpt_path, ROOT),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RES_DIR, f"{TAG}_classification_report.txt"), "w") as f:
        f.write(classification_report(
            y_true, y_pred, labels=labels_idx, target_names=class_names, zero_division=0,
        ))

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {CFG['label']} s{SEED}")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{TAG}_cm.png"))
    plt.close()

    print(f"\nTest Accuracy: {test_acc:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved: {METRICS_JSON}")

print(f"\nDone. Artifacts in compare/ for {TAG} ({STAGE}).")
