import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
args = parser.parse_args()

SEED = args.seed
RUN_NAME = f"_{args.run_name}" if args.run_name else ""

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "ProcessedDataset224_PublicLocal"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 80
AUTOTUNE = tf.data.AUTOTUNE

print("=" * 60)
print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Valid", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

print(f"\nClasses: {class_names}")

custom_weights = {
    0: 2.0,
    1: 2.5,
    2: 1.0,
    3: 1.3,
}
class_weights = custom_weights
print(f"Custom class weights: {class_weights}")
alpha_tensor = tf.constant([class_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

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

# ── Hard-label CutMix ──
def apply_cutmix(images, labels, beta=0.5, mix_prob=0.5):
    batch_size = tf.shape(images)[0]
    img_h = tf.shape(images)[1]
    img_w = tf.shape(images)[2]
    do_mix = tf.random.uniform(()) < mix_prob

    def _cutmix():
        lam = tf.random.uniform((), minval=0.1, maxval=0.9)
        lam = tf.maximum(lam, 1.0 - lam)
        r = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(tf.cast(img_h, tf.float32) * r, tf.int32)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * r, tf.int32)

        cx = tf.random.uniform((), maxval=img_w, dtype=tf.int32)
        cy = tf.random.uniform((), maxval=img_h, dtype=tf.int32)

        x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

        idx = tf.random.shuffle(tf.range(batch_size))
        img2 = tf.gather(images, idx)
        lbl2 = tf.gather(labels, idx)

        y_range = tf.range(img_h)
        x_range = tf.range(img_w)
        Y, X = tf.meshgrid(y_range, x_range, indexing='ij')
        mask = tf.cast((Y >= y1) & (Y < y2) & (X >= x1) & (X < x2), tf.float32)
        mask = tf.expand_dims(mask, -1)
        mask = tf.expand_dims(mask, 0)
        mixed = images * (1.0 - mask) + img2 * mask

        box_pixels = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
        total_pixels = tf.cast(img_h * img_w, tf.float32)
        lam_actual = 1.0 - box_pixels / total_pixels

        mixed_lbl = tf.where(lam_actual >= 0.5, labels, lbl2)
        return mixed, mixed_lbl

    return tf.cond(do_mix, _cutmix, lambda: (images, labels))

# ── Base augmentation ──
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.05),
])

def augment(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_aug(images, training=True)
    images = tf.clip_by_value(images, 0, 255)
    images = images / 255.0
    images = tf.image.random_hue(images, max_delta=0.05)
    images = tf.image.random_saturation(images, lower=0.7, upper=1.3)
    images, labels = apply_cutmix(images, labels, beta=0.5, mix_prob=0.5)
    images = images * 255.0
    return images, labels

class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, start_ratio=0.6):
        super().__init__()
        self.start_ratio = start_ratio
        self.swa_weights = None
        self.swa_count = 0
        self.start_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            total = self.params.get("epochs", 80)
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

print(f"\nLoading best model from phase 1...")
model = keras.models.load_model(
    f"models/best_model_effnet{RUN_NAME}.keras",
    custom_objects={"loss_fn": focal_loss(alpha=alpha_tensor.numpy())}
)

base_model = model.get_layer("efficientnetb0")
base_model.trainable = True

for layer in base_model.layers[:]:
    layer.trainable = False

for layer in base_model.layers[-35:]:
    layer.trainable = True
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_layers = [l.name for l in base_model.layers if l.trainable]
print(f"\nUnfrozen layers ({len(trainable_layers)}): {trainable_layers}")
print(f"Total EfficientNetB0 layers: {len(base_model.layers)}")

bn_trainable = sum(1 for l in base_model.layers[-35:] if isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable)
bn_frozen = sum(1 for l in base_model.layers[-35:] if isinstance(l, tf.keras.layers.BatchNormalization) and not l.trainable)
print(f"BN layers in top 35: {bn_trainable} trainable, {bn_frozen} frozen")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
    metrics=["accuracy"]
)

train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_ds_p = valid_ds.prefetch(AUTOTUNE)
test_ds_p = test_ds.prefetch(AUTOTUNE)

print("\n" + "=" * 60)
print("Starting EfficientNetB0 CutMix fine-tune (35 layers, BN frozen)...")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(f"models/fine_tuned_best_cutmix{RUN_NAME}.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    CSVLogger(f"models/fine_tuning_log_cutmix{RUN_NAME}.csv"),
]

history = model.fit(
    train_ds_aug,
    validation_data=valid_ds_p,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f"\nLoading best fine-tuned model...")
model = keras.models.load_model(
    f"models/fine_tuned_best_cutmix{RUN_NAME}.keras",
    custom_objects={"loss_fn": focal_loss(alpha=alpha_tensor.numpy())}
)

print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_ds_p, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

model.save(f"models/fine_tuned_cutmix_effnet{RUN_NAME}.keras")

y_true, y_pred = [], []
for images, labels in test_ds_p:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, colorbar=False)
plt.title(f"CutMix EffNet Confusion{RUN_NAME}")
plt.savefig(f"models/fine_tuned_cm_effnet{RUN_NAME}.png")
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Valid")
ax1.set_title("CutMix Fine-Tuned Accuracy")
ax1.legend()
ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Valid")
ax2.set_title("CutMix Fine-Tuned Loss")
ax2.legend()
plt.savefig(f"models/fine_tuned_curves_effnet{RUN_NAME}.png")
plt.close()

print(f"\nCutMix fine-tuning complete!{' (run ' + args.run_name + ')' if args.run_name else ''}")
