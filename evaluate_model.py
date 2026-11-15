import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

model = keras.models.load_model("models/best_model.keras")

TEST_PATH = "ProcessedDataset224_PublicLocal/Test"
IMAGE_SIZE = (224, 224)

class_names = sorted([d for d in os.listdir(TEST_PATH) if os.path.isdir(os.path.join(TEST_PATH, d))])
print("Classes:", class_names)

image_paths = []
labels = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(TEST_PATH, class_name)
    for image_name in sorted(os.listdir(class_folder)):
        image_paths.append(os.path.join(class_folder, image_name))
        labels.append(class_index)

print(f"Total test images: {len(image_paths)}")

batch_size = 32
all_preds = []
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch = []
    for p in batch_paths:
        img = keras.utils.load_img(p, target_size=IMAGE_SIZE)
        img_array = keras.utils.img_to_array(img)
        batch.append(img_array)
    batch = np.array(batch)
    preds = model.predict(batch, verbose=0)
    all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)
y_pred = np.argmax(all_preds, axis=1)
y_true = np.array(labels)

accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (rows=true, cols=pred):")
print("         ", end="")
for c in class_names:
    print(f"{c:>10}", end="")
print()
for i, c in enumerate(class_names):
    print(f"{c:<10}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i,j]:>10}", end="")
    print()

print("\nPer-class accuracy:")
for i, c in enumerate(class_names):
    correct = cm[i,i]
    total = cm[i].sum()
    print(f"  {c:<12} {correct}/{total} = {correct/total*100:.2f}%")

print("\nMisclassified:")
for i, (p, path) in enumerate(zip(y_pred, image_paths)):
    if p != y_true[i]:
        img_name = os.path.basename(path)
        print(f"  {img_name:35s} True: {class_names[y_true[i]]:12s} Pred: {class_names[p]:12s} Conf: {all_preds[i][p]*100:.2f}%")

plt.figure(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Evaluation Confusion Matrix")
plt.savefig("models/evaluation_confusion_matrix.png")
print("\nConfusion matrix saved to models/evaluation_confusion_matrix.png")