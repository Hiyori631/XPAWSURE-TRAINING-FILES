"""Stratified 5-fold ensemble training."""
import os, json, random, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

DATASET_PATH = "ProcessedDataset224_PublicLocal"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 80
N_FOLDS = 5
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs

print("Loading all training data...")
full_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
class_names = full_ds.class_names
NUM_CLASSES = len(class_names)

print(f"Classes: {class_names}")

image_paths = []
labels = []
for cls in class_names:
    cls_dir = os.path.join(f"{DATASET_PATH}/Train", cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            image_paths.append(os.path.join(cls_dir, fname))
            labels.append(class_names.index(cls))

image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"Total training samples: {len(image_paths)}")

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img, label

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}/{N_FOLDS}")
    print(f"{'='*60}")

    seed = 42 + fold
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_paths_fold = image_paths[train_idx]
    train_labels_fold = labels[train_idx]
    val_paths_fold = image_paths[val_idx]
    val_labels_fold = labels[val_idx]

    print(f"  Train samples: {len(train_paths_fold)}")
    print(f"  Val samples:   {len(val_paths_fold)}")

    train_ds_fold = tf.data.Dataset.from_tensor_slices((train_paths_fold, train_labels_fold))
    train_ds_fold = train_ds_fold.map(load_image, num_parallel_calls=AUTOTUNE)
    train_ds_fold = train_ds_fold.shuffle(1000, seed=seed).batch(BATCH_SIZE)

    val_ds_fold = tf.data.Dataset.from_tensor_slices((val_paths_fold, val_labels_fold))
    val_ds_fold = val_ds_fold.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels_fold), y=train_labels_fold
    )
    class_weights = dict(enumerate(class_weights))
    alpha_tensor = tf.constant([class_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

    total_steps = EPOCHS * len(train_ds_fold)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-5
    )

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ])

    def augment(images, labels):
        images = tf.cast(images, tf.float32)
        images = data_aug(images, training=True)
        images = tf.clip_by_value(images, 0, 255)
        return images, labels

    train_ds_fold = train_ds_fold.map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds_fold = val_ds_fold.prefetch(AUTOTUNE)

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

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
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

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ModelCheckpoint(f"models/fold_{fold}.keras", monitor="val_accuracy", mode="max", save_best_only=True),
        CSVLogger(f"models/fold_{fold}_log.csv")
    ]

    model.fit(
        train_ds_fold, validation_data=val_ds_fold,
        epochs=EPOCHS, callbacks=callbacks, verbose=1
    )

    print(f"Fold {fold + 1} complete. Model saved to models/fold_{fold}.keras")

print("\n" + "=" * 60)
print("All 5 folds complete!")
print("=" * 60)
print("Models saved:")
for fold in range(N_FOLDS):
    path = f"models/fold_{fold}.keras"
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path} ({size:.1f} MB)")