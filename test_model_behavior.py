"""
Test screening thresholds against actual model behavior.
Loads the real Keras model and runs predictions on disease images + synthetic negatives.
Records confidence, margin, entropy, and feature similarity for each image.

Usage:
    python test_model_behavior.py
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'fine_tuned_best_effnet_clean_5class_s42.keras')
CLASS_NAMES_PATH = os.path.join(SCRIPT_DIR, 'models', 'class_names_clean_5class_s42.json')
TEST_DATASET = os.path.join(SCRIPT_DIR, 'ProcessedDataset224_Clean_5class', 'Test')
TRAINING_REF_PATH = os.path.join(SCRIPT_DIR, 'models', 'training_reference.json')

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

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
print("SCREENING THRESHOLD VALIDATION")
print("=" * 70)
print(f"Loading model from: {KERAS_MODEL_PATH}")

model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    custom_objects={"loss_fn": focal_loss(gamma=2.0, alpha=alpha.numpy())}
)
print(f"Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")

print("Creating feature extractor for global_average_pooling2d layer...")
feature_layer = model.get_layer('global_average_pooling2d')
feature_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer.output)
print(f"Feature extractor created. Output shape: {feature_model.output_shape}")

with open(TRAINING_REF_PATH) as f:
    training_ref = json.load(f)
training_mean = np.array(training_ref['mean_feature'], dtype=np.float32)
print(f"Loaded training reference: {training_ref['num_images']} images, {training_ref['feature_dim']} features")

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-8)

THRESHOLDS = {
    'DISEASE_CONFIDENCE_MIN': 0.25,
    'CONFIDENCE_MARGIN_MIN': 0.08,
    'ENTROPY_MAX': 1.4,
    'FEATURE_SIMILARITY_MIN': 0.15,
}

def shannon_entropy(probs):
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def classify_result(top1, margin, entropy, similarity):
    if similarity < THRESHOLDS['FEATURE_SIMILARITY_MIN']:
        return 'UNCERTAIN', f'similarity={similarity:.3f} < {THRESHOLDS["FEATURE_SIMILARITY_MIN"]}'
    if top1 < THRESHOLDS['DISEASE_CONFIDENCE_MIN']:
        return 'UNCERTAIN', f'top1={top1:.3f} < {THRESHOLDS["DISEASE_CONFIDENCE_MIN"]}'
    if margin < THRESHOLDS['CONFIDENCE_MARGIN_MIN']:
        return 'UNCERTAIN', f'margin={margin:.3f} < {THRESHOLDS["CONFIDENCE_MARGIN_MIN"]}'
    if entropy > THRESHOLDS['ENTROPY_MAX']:
        return 'UNCERTAIN', f'entropy={entropy:.3f} > {THRESHOLDS["ENTROPY_MAX"]}'
    return 'DISEASE_DETECTED', ''

def predict_image(path, label):
    try:
        img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        logits = model.predict(arr, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]

        features = feature_model.predict(arr, verbose=0)
        feature_vec = np.array(features).flatten()
        similarity = float(cosine_similarity(feature_vec, training_mean))

        sorted_idx = np.argsort(probs)[::-1]
        top1 = float(probs[sorted_idx[0]])
        top2 = float(probs[sorted_idx[1]])
        margin = top1 - top2
        entropy = shannon_entropy(probs)
        state, reason = classify_result(top1, margin, entropy, similarity)

        mean_brightness = float(np.mean(arr))
        variance = float(np.var(arr))

        return {
            'label': label,
            'top_class': CLASS_NAMES[sorted_idx[0]],
            'top1': top1,
            'top2_class': CLASS_NAMES[sorted_idx[1]],
            'top2': top2,
            'margin': margin,
            'entropy': entropy,
            'similarity': similarity,
            'state': state,
            'reason': reason,
            'brightness': mean_brightness,
            'variance': variance,
            'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(5)},
            'status': 'OK'
        }
    except Exception as e:
        return {'label': label, 'status': 'ERROR', 'error': str(e)}

all_results = []

print("\n" + "=" * 70)
print("PART 1: DISEASE IMAGES (from test set)")
print("=" * 70)

for disease in os.listdir(TEST_DATASET):
    disease_dir = os.path.join(TEST_DATASET, disease)
    if not os.path.isdir(disease_dir):
        continue
    images = [f for f in os.listdir(disease_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        continue
    for img_file in images[:5]:
        img_path = os.path.join(disease_dir, img_file)
        result = predict_image(img_path, f"{disease}/{img_file}")
        all_results.append(result)
        if result['status'] == 'OK':
            icon = '  ' if result['state'] == 'DISEASE_DETECTED' else '!!'
            print(f"  {icon} [{result['label']:30s}] Top1={result['top1']*100:5.1f}% {result['top_class']:20s} Margin={result['margin']*100:5.1f}% Entropy={result['entropy']:.3f} -> {result['state']}")

print("\n" + "=" * 70)
print("PART 2: SYNTHETIC NEGATIVE IMAGES")
print("=" * 70)

synthetic_tests = {
    'black_image': np.zeros((224, 224, 3), dtype=np.float32),
    'white_image': np.ones((224, 224, 3), dtype=np.float32) * 255,
    'gray_image': np.ones((224, 224, 3), dtype=np.float32) * 128,
    'dark_gray': np.ones((224, 224, 3), dtype=np.float32) * 30,
    'bright_gray': np.ones((224, 224, 3), dtype=np.float32) * 230,
    'random_noise': np.random.randint(0, 256, (224, 224, 3)).astype(np.float32),
    'red_only': np.zeros((224, 224, 3), dtype=np.float32),
    'gradient_h': np.tile(np.linspace(0, 255, 224).reshape(1, 224, 1), (224, 1, 3)).astype(np.float32),
    'gradient_v': np.tile(np.linspace(0, 255, 224).reshape(224, 1, 1), (1, 224, 3)).astype(np.float32),
    'checkerboard': np.zeros((224, 224, 3), dtype=np.float32),
}

for i in range(224):
    for j in range(224):
        synthetic_tests['red_only'][i, j, 0] = 200
        synthetic_tests['red_only'][i, j, 1] = 50
        synthetic_tests['red_only'][i, j, 2] = 50
        synthetic_tests['checkerboard'][i, j] = 255 if (i // 32 + j // 32) % 2 == 0 else 0

for name, arr in synthetic_tests.items():
    try:
        arr_batch = np.expand_dims(arr, axis=0)
        logits = model.predict(arr_batch, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]

        features = feature_model.predict(arr_batch, verbose=0)
        feature_vec = np.array(features).flatten()
        similarity = float(cosine_similarity(feature_vec, training_mean))

        sorted_idx = np.argsort(probs)[::-1]
        top1 = float(probs[sorted_idx[0]])
        top2 = float(probs[sorted_idx[1]])
        margin = top1 - top2
        entropy = shannon_entropy(probs)
        state, reason = classify_result(top1, margin, entropy, similarity)

        mean_brightness = float(np.mean(arr))
        variance = float(np.var(arr))

        result = {
            'label': name,
            'top_class': CLASS_NAMES[sorted_idx[0]],
            'top1': top1,
            'top2_class': CLASS_NAMES[sorted_idx[1]],
            'top2': top2,
            'margin': margin,
            'entropy': entropy,
            'similarity': similarity,
            'state': state,
            'reason': reason,
            'brightness': mean_brightness,
            'variance': variance,
            'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(5)},
            'status': 'OK'
        }
        all_results.append(result)
        icon = '  ' if state == 'DISEASE_DETECTED' else '!!'
        print(f"  {icon} [{name:30s}] Top1={top1*100:5.1f}% {CLASS_NAMES[sorted_idx[0]]:20s} Margin={margin*100:5.1f}% Entropy={entropy:.3f} Sim={similarity:.3f} -> {state}")

    except Exception as e:
        print(f"  ERROR [{name}]: {e}")
        all_results.append({'label': name, 'status': 'ERROR', 'error': str(e)})

print("\n" + "=" * 70)
print("PART 3: EXTENDED NEGATIVE IMAGES (realistic objects)")
print("=" * 70)

np.random.seed(42)

extended_tests = {}

brown_base = np.full((224, 224, 3), [139, 90, 43], dtype=np.float32)
extended_tests['solid_brown'] = brown_base.copy()

noise_small = np.random.normal(0, 10, (224, 224, 3))
extended_tests['brown_textured'] = np.clip(brown_base + noise_small, 0, 255)

skin = np.zeros((224, 224, 3), dtype=np.float32)
skin[:,:,0] = np.random.uniform(175, 215, (224, 224))
skin[:,:,1] = np.random.uniform(135, 175, (224, 224))
skin[:,:,2] = np.random.uniform(95, 135, (224, 224))
extended_tests['human_skin_tone'] = skin

leather_base = np.full((224, 224, 3), [101, 67, 33], dtype=np.float32)
wrinkles = np.zeros((224, 224), dtype=np.float32)
for k in range(8):
    freq = 0.1 + k * 0.05
    phase = np.random.uniform(0, 2 * np.pi)
    wrinkles += np.sin(np.linspace(0, freq * 2 * np.pi, 224).reshape(-1, 1) + phase) * (3 - k * 0.3)
wrinkles = np.clip(wrinkles, -10, 10)
leather_noise = np.random.normal(0, 5, (224, 224, 3))
extended_tests['leather_texture'] = np.clip(leather_base + wrinkles.reshape(224, 224, 1) + leather_noise, 0, 255)

wood_base = np.zeros((224, 224, 3), dtype=np.float32)
for i in range(224):
    base_val = 120 + 40 * np.sin(i * 0.03) + np.random.normal(0, 8)
    wood_base[i, :, 0] = base_val + 20
    wood_base[i, :, 1] = base_val
    wood_base[i, :, 2] = base_val - 15
for j in range(224):
    grain = np.sin(j * 0.8) * 5 + np.sin(j * 2.3) * 3
    wood_base[:, j, :] += grain
extended_tests['wooden_surface'] = np.clip(wood_base, 0, 255)

beige_base = np.full((224, 224, 3), [210, 200, 180], dtype=np.float32)
beige_noise = np.random.normal(0, 3, (224, 224, 3))
extended_tests['beige_wall'] = np.clip(beige_base + beige_noise, 0, 255)

fur_base = np.full((224, 224, 3), [180, 150, 100], dtype=np.float32)
fur_noise = np.random.normal(0, 20, (224, 224, 3))
extended_tests['dog_fur_healthy'] = np.clip(fur_base + fur_noise, 0, 255)

fabric_base = np.full((224, 224, 3), [100, 100, 120], dtype=np.float32)
for i in range(224):
    for j in range(224):
        if (i % 4 < 2) != (j % 4 < 2):
            fabric_base[i, j] += 15
fabric_noise = np.random.normal(0, 5, (224, 224, 3))
extended_tests['fabric_texture'] = np.clip(fabric_base + fabric_noise, 0, 255)

orange_base = np.full((224, 224, 3), [210, 120, 50], dtype=np.float32)
orange_noise = np.random.normal(0, 12, (224, 224, 3))
extended_tests['orange_surface'] = np.clip(orange_base + orange_noise, 0, 255)

black_leather = np.full((224, 224, 3), [30, 30, 35], dtype=np.float32)
black_noise = np.random.normal(0, 8, (224, 224, 3))
wrinkles2 = np.zeros((224, 224), dtype=np.float32)
for k in range(5):
    freq = 0.08 + k * 0.04
    phase = np.random.uniform(0, 2 * np.pi)
    wrinkles2 += np.sin(np.linspace(0, freq * 2 * np.pi, 224).reshape(-1, 1) + phase) * (2 - k * 0.3)
extended_tests['black_leather'] = np.clip(black_leather + wrinkles2.reshape(224, 224, 1) + black_noise, 0, 255)

cat_fur = np.full((224, 224, 3), [160, 130, 90], dtype=np.float32)
cat_noise = np.random.normal(0, 25, (224, 224, 3))
extended_tests['cat_fur'] = np.clip(cat_fur + cat_noise, 0, 255)

carpet_base = np.full((224, 224, 3), [80, 60, 50], dtype=np.float32)
for i in range(224):
    for j in range(224):
        if (i // 3 + j // 3) % 2 == 0:
            carpet_base[i, j] += 20
carpet_noise = np.random.normal(0, 10, (224, 224, 3))
extended_tests['carpet'] = np.clip(carpet_base + carpet_noise, 0, 255)

concrete_base = np.full((224, 224, 3), [160, 160, 155], dtype=np.float32)
concrete_noise = np.random.normal(0, 15, (224, 224, 3))
extended_tests['concrete_floor'] = np.clip(concrete_base + concrete_noise, 0, 255)

for name, arr in extended_tests.items():
    try:
        arr_batch = np.expand_dims(arr, axis=0)
        logits = model.predict(arr_batch, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]

        features = feature_model.predict(arr_batch, verbose=0)
        feature_vec = np.array(features).flatten()
        similarity = float(cosine_similarity(feature_vec, training_mean))

        sorted_idx = np.argsort(probs)[::-1]
        top1 = float(probs[sorted_idx[0]])
        top2 = float(probs[sorted_idx[1]])
        margin = top1 - top2
        entropy = shannon_entropy(probs)
        state, reason = classify_result(top1, margin, entropy, similarity)

        mean_brightness = float(np.mean(arr))
        variance = float(np.var(arr))

        result = {
            'label': name,
            'top_class': CLASS_NAMES[sorted_idx[0]],
            'top1': top1,
            'top2_class': CLASS_NAMES[sorted_idx[1]],
            'top2': top2,
            'margin': margin,
            'entropy': entropy,
            'similarity': similarity,
            'state': state,
            'reason': reason,
            'brightness': mean_brightness,
            'variance': variance,
            'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(5)},
            'status': 'OK'
        }
        all_results.append(result)
        icon = '  ' if state == 'DISEASE_DETECTED' else '!!'
        print(f"  {icon} [{name:30s}] Top1={top1*100:5.1f}% {CLASS_NAMES[sorted_idx[0]]:20s} Margin={margin*100:5.1f}% Entropy={entropy:.3f} Sim={similarity:.3f} -> {state}")

    except Exception as e:
        print(f"  ERROR [{name}]: {e}")
        all_results.append({'label': name, 'status': 'ERROR', 'error': str(e)})

print("\n" + "=" * 70)
print("PART 4: SUMMARY")
print("=" * 70)

ok_results = [r for r in all_results if r['status'] == 'OK']
disease_results = [r for r in ok_results if '/' in r['label']]
synthetic_results = [r for r in ok_results if '/' not in r['label']]
extended_results = [r for r in ok_results if r['label'] in extended_tests]

print(f"\nTotal tested: {len(ok_results)} images")
print(f"  Disease images: {len(disease_results)}")
print(f"  Basic synthetic negatives: {len(synthetic_results) - len(extended_results)}")
print(f"  Extended negatives (realistic): {len(extended_results)}")

disease_detected = sum(1 for r in disease_results if r['state'] == 'DISEASE_DETECTED')
disease_uncertain = sum(1 for r in disease_results if r['state'] == 'UNCERTAIN')
print(f"\nDisease images:")
print(f"  DISEASE_DETECTED: {disease_detected}/{len(disease_results)} ({disease_detected/len(disease_results)*100:.0f}%)")
print(f"  UNCERTAIN (false reject): {disease_uncertain}/{len(disease_results)} ({disease_uncertain/len(disease_results)*100:.0f}%)")
if disease_uncertain > 0:
    print(f"  Rejected images:")
    for r in disease_results:
        if r['state'] == 'UNCERTAIN':
            print(f"    {r['label']:30s} Top1={r['top1']*100:.1f}% Margin={r['margin']*100:.1f}% Entropy={r['entropy']:.3f} Reason: {r['reason']}")

basic_synth = [r for r in synthetic_results if r['label'] not in extended_tests]
basic_detected = sum(1 for r in basic_synth if r['state'] == 'DISEASE_DETECTED')
basic_uncertain = sum(1 for r in basic_synth if r['state'] == 'UNCERTAIN')
print(f"\nBasic synthetic negatives:")
print(f"  DISEASE_DETECTED (false positive): {basic_detected}/{len(basic_synth)} ({basic_detected/len(basic_synth)*100:.0f}%)")
print(f"  UNCERTAIN (correctly rejected): {basic_uncertain}/{len(basic_synth)} ({basic_uncertain/len(basic_synth)*100:.0f}%)")

ext_detected = sum(1 for r in extended_results if r['state'] == 'DISEASE_DETECTED')
ext_uncertain = sum(1 for r in extended_results if r['state'] == 'UNCERTAIN')
print(f"\nExtended negatives (realistic objects):")
print(f"  DISEASE_DETECTED (false positive): {ext_detected}/{len(extended_results)} ({ext_detected/len(extended_results)*100:.0f}%)")
print(f"  UNCERTAIN (correctly rejected): {ext_uncertain}/{len(extended_results)} ({ext_uncertain/len(extended_results)*100:.0f}%)")
if ext_detected > 0:
    print(f"  MISSED images (showed as disease):")
    for r in extended_results:
        if r['state'] == 'DISEASE_DETECTED':
            print(f"    {r['label']:30s} Top1={r['top1']*100:.1f}% {r['top_class']} Margin={r['margin']*100:.1f}%")

all_negatives = basic_synth + extended_results
all_detected = sum(1 for r in all_negatives if r['state'] == 'DISEASE_DETECTED')
all_uncertain = sum(1 for r in all_negatives if r['state'] == 'UNCERTAIN')
print(f"\n{'='*70}")
print(f"ALL NEGATIVES COMBINED:")
print(f"  DISEASE_DETECTED (false positive): {all_detected}/{len(all_negatives)} ({all_detected/len(all_negatives)*100:.0f}%)")
print(f"  UNCERTAIN (correctly rejected): {all_uncertain}/{len(all_negatives)} ({all_uncertain/len(all_negatives)*100:.0f}%)")

if ok_results:
    all_top1 = [r['top1'] for r in ok_results]
    all_margins = [r['margin'] for r in ok_results]
    all_entropies = [r['entropy'] for r in ok_results]
    all_similarities = [r['similarity'] for r in ok_results]
    print(f"\nOverall statistics:")
    print(f"  Top1 confidence: min={min(all_top1)*100:.1f}% max={max(all_top1)*100:.1f}% mean={np.mean(all_top1)*100:.1f}%")
    print(f"  Margin:          min={min(all_margins)*100:.1f}% max={max(all_margins)*100:.1f}% mean={np.mean(all_margins)*100:.1f}%")
    print(f"  Entropy:         min={min(all_entropies):.3f} max={max(all_entropies):.3f} mean={np.mean(all_entropies):.3f}")
    print(f"  Similarity:      min={min(all_similarities):.3f} max={max(all_similarities):.3f} mean={np.mean(all_similarities):.3f}")

print(f"\nThresholds used:")
print(f"  DISEASE_CONFIDENCE_MIN: {THRESHOLDS['DISEASE_CONFIDENCE_MIN']*100:.0f}%")
print(f"  CONFIDENCE_MARGIN_MIN:  {THRESHOLDS['CONFIDENCE_MARGIN_MIN']*100:.0f}%")
print(f"  ENTROPY_MAX:            {THRESHOLDS['ENTROPY_MAX']}")
print(f"  FEATURE_SIMILARITY_MIN: {THRESHOLDS['FEATURE_SIMILARITY_MIN']}")
