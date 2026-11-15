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
EPOCHS_STAGE1 = 80
EPOCHS_STAGE2 = 50
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

def freeze_layer_config(base_model, unfreeze_last=35):
    for layer in base_model.layers[:]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_last:]:
        layer.trainable = True
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    bn_trainable = sum(1 for l in base_model.layers[-unfreeze_last:] if isinstance(l, tf.keras.layers.BatchNormalization) and l.trainable)
    bn_frozen = sum(1 for l in base_model.layers[-unfreeze_last:] if isinstance(l, tf.keras.layers.BatchNormalization) and not l.trainable)
    trainable_names = [l.name for l in base_model.layers if l.trainable]
    return trainable_count, trainable_names, bn_trainable, bn_frozen

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
    images = images * 255.0
    return images, labels

train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_ds_p = valid_ds.prefetch(AUTOTUNE)
test_ds_p = test_ds.prefetch(AUTOTUNE)

# ============================================================
# STAGE 1: 35 layers @ 1e-5
# ============================================================
print("\n" + "=" * 60)
print(f"STAGE 1: Loading frozen model (best_model_effnet{RUN_NAME}.keras)")
print("=" * 60)

model = keras.models.load_model(
    f"models/best_model_effnet{RUN_NAME}.keras",
    custom_objects={"loss_fn": focal_loss(alpha=alpha_tensor.numpy())}
)

base_model = model.get_layer("efficientnetb0")
base_model.trainable = True

tc1, tn1, bnt1, bnf1 = freeze_layer_config(base_model, unfreeze_last=35)
print(f"\nStage 1: {tc1} unfrozen layers, BN: {bnt1} trainable / {bnf1} frozen")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
    metrics=["accuracy"]
)

callbacks_s1 = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(f"models/stage1_effnet{RUN_NAME}.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    CSVLogger(f"models/stage1_log_effnet{RUN_NAME}.csv"),
    SWACallback(start_ratio=0.7),
]

print("\n--- Stage 1 training ---")
history1 = model.fit(
    train_ds_aug,
    validation_data=valid_ds_p,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_s1,
    verbose=1
)

# ============================================================
# STAGE 2: 50 layers @ 1e-6 with CosineDecayRestarts
# ============================================================
print("\n" + "=" * 60)
print(f"STAGE 2: Loading best Stage 1 model (stage1_effnet{RUN_NAME}.keras)")
print("=" * 60)

model = keras.models.load_model(
    f"models/stage1_effnet{RUN_NAME}.keras",
    custom_objects={"loss_fn": focal_loss(alpha=alpha_tensor.numpy())}
)

base_model = model.get_layer("efficientnetb0")
base_model.trainable = True

tc2, tn2, bnt2, bnf2 = freeze_layer_config(base_model, unfreeze_last=50)
print(f"\nStage 2: {tc2} unfrozen layers, BN: {bnt2} trainable / {bnf2} frozen")

total_steps_s2 = EPOCHS_STAGE2 * len(train_ds)
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-6,
    first_decay_steps=total_steps_s2 // 3,
    t_mul=2.0,
    m_mul=0.5,
    alpha=1e-7
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
    metrics=["accuracy"]
)

callbacks_s2 = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(f"models/fine_tuned_best_twostage{RUN_NAME}.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    CSVLogger(f"models/stage2_log_effnet{RUN_NAME}.csv"),
    SWACallback(start_ratio=0.7),
]

print("\n--- Stage 2 training (50 layers, CosineDecayRestarts, 1e-6) ---")
history2 = model.fit(
    train_ds_aug,
    validation_data=valid_ds_p,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_s2,
    verbose=1
)

# ============================================================
# Final evaluation
# ============================================================
print(f"\nLoading best fine-tuned model...")
model = keras.models.load_model(
    f"models/fine_tuned_best_twostage{RUN_NAME}.keras",
    custom_objects={"loss_fn": focal_loss(alpha=alpha_tensor.numpy())}
)

print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_ds_p, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

model.save(f"models/fine_tuned_two_stage_effnet{RUN_NAME}.keras")

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
plt.title(f"Two-Stage EffNet Confusion{RUN_NAME}")
plt.savefig(f"models/fine_tuned_cm_effnet{RUN_NAME}.png")
plt.close()

print(f"\nTwo-stage fine-tuning complete!{' (run ' + args.run_name + ')' if args.run_name else ''}")
