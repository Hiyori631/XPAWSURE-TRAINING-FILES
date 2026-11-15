import os, sys, json, random, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--dataset", type=str, default="ProcessedDataset224_PublicLocal")
args = parser.parse_args()

SEED = args.seed
RUN_NAME = f"_{args.run_name}" if args.run_name else ""

tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = args.dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE
os.makedirs("models", exist_ok=True)

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
print(f"Class count: {NUM_CLASSES}")

custom_weights = {
    0: 3.0,  # Allergic Dermatitis (fewest samples)
    1: 2.0,  # Bacterial
    2: 2.5,  # Fungal
    3: 1.0,  # Hotspot
    4: 1.3,  # Mange
}
class_weights = custom_weights
print(f"\nCustom class weights: {class_weights}")

with open(f"models/class_names{RUN_NAME}.json", "w") as f:
    json.dump(class_names, f)
print(f"Saved class_names.json: {class_names}")

total_steps = EPOCHS * len(train_ds)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-5
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
])

def augment(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_augmentation(images, training=True)
    images = tf.clip_by_value(images, 0, 255)
    images = images / 255.0
    images = tf.image.random_hue(images, max_delta=0.05)
    images = tf.image.random_saturation(images, lower=0.7, upper=1.3)
    images = images * 255.0
    return images, labels

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

alpha_tensor = tf.constant([class_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss(gamma=2.0, alpha=alpha_tensor),
    metrics=["accuracy"]
)

model.summary()

class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, start_ratio=0.7):
        super().__init__()
        self.start_ratio = start_ratio
        self.swa_weights = None
        self.swa_count = 0
        self.start_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            total = self.params.get("epochs", 100)
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

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(f"models/best_model_effnet{RUN_NAME}.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    CSVLogger(f"models/training_log_effnet{RUN_NAME}.csv"),
    SWACallback(start_ratio=0.7),
]

train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("\n" + "=" * 60)
print("Starting EfficientNetB0 training (frozen backbone)...")
print("=" * 60)

history = model.fit(
    train_ds_aug,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nLoading best model...")
model = models.load_model(f"models/best_model_effnet{RUN_NAME}.keras", custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha_tensor.numpy())})

print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

model.save(f"models/final_model_effnet{RUN_NAME}.keras")

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax, colorbar=False)
plt.title(f"Confusion Matrix - EffNet{RUN_NAME}")
plt.savefig(f"models/confusion_matrix_effnet{RUN_NAME}.png")
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Valid")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Valid")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.savefig(f"models/training_curves_effnet{RUN_NAME}.png")
plt.close()

print(f"\nDone! Models saved to models/")
