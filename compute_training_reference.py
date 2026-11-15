"""
Compute training set reference statistics for OOD detection.
Extracts features from global_average_pooling2d layer for all training images
and computes the mean feature vector.

Usage:
    python compute_training_reference.py
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
TRAIN_DATASET = os.path.join(SCRIPT_DIR, 'ProcessedDataset224_Clean_5class', 'Train')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'models', 'training_reference.json')

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

print("=" * 70)
print("COMPUTE TRAINING REFERENCE FOR OOD DETECTION")
print("=" * 70)

print(f"Loading model from: {KERAS_MODEL_PATH}")
model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)
print(f"Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")

print(f"\nCreating feature extractor for global_average_pooling2d layer...")
feature_layer = model.get_layer('global_average_pooling2d')
feature_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)
print(f"Feature extractor created. Output shape: {feature_model.output_shape}")

print(f"\nProcessing training images from: {TRAIN_DATASET}")
all_features = []
all_labels = []

for disease in os.listdir(TRAIN_DATASET):
    disease_dir = os.path.join(TRAIN_DATASET, disease)
    if not os.path.isdir(disease_dir):
        continue
    
    images = [f for f in os.listdir(disease_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n  {disease}: {len(images)} images")
    
    for i, img_file in enumerate(images):
        img_path = os.path.join(disease_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)
            
            features = feature_model.predict(arr, verbose=0)
            feature_vec = np.array(features).flatten()
            
            all_features.append(feature_vec)
            all_labels.append(disease)
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(images)} images...")
        except Exception as e:
            print(f"    ERROR processing {img_file}: {e}")

all_features = np.array(all_features)
print(f"\nTotal features extracted: {all_features.shape}")
print(f"Feature dimension: {all_features.shape[1]}")

mean_feature = np.mean(all_features, axis=0)
std_feature = np.std(all_features, axis=0)

print(f"\nFeature statistics:")
print(f"  Mean: min={mean_feature.min():.4f} max={mean_feature.max():.4f}")
print(f"  Std:  min={std_feature.min():.4f} max={std_feature.max():.4f}")

reference = {
    'mean_feature': mean_feature.tolist(),
    'std_feature': std_feature.tolist(),
    'feature_dim': int(all_features.shape[1]),
    'num_images': int(all_features.shape[0]),
    'classes': list(set(all_labels)),
    'description': 'Mean and std of global_average_pooling2d features from training set for OOD detection'
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\nTraining reference saved to: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH)} bytes")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
