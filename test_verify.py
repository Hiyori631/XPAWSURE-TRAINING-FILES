"""Quick pipeline verification - runs 2 epochs to catch runtime errors."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("Loading dataset...")
DATASET_PATH = "ProcessedDataset224_PublicLocal"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
class_names = train_ds.class_names
print(f"Classes: {class_names}")
print(f"Found {len(train_ds)} train batches")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Valid", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
print(f"Found {len(valid_ds)} valid batches")

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/Test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
)
print(f"Found {len(test_ds)} test batches")

# Build model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train for 1 epoch
model.fit(train_ds, validation_data=valid_ds, epochs=1, verbose=1)

# Evaluate
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")
print("Pipeline verification PASSED")