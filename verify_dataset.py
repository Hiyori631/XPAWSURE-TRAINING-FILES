import os
from PIL import Image

DATASET_PATH = "ProcessedDataset224_PublicLocal"

splits = ["Train", "Valid", "Test"]

print("=" * 60)

total_images = 0
corrupted = []

for split in splits:
    print(f"\n{split.upper()}")

    split_path = os.path.join(DATASET_PATH, split)

    for disease in sorted(os.listdir(split_path)):

        disease_path = os.path.join(split_path, disease)

        if not os.path.isdir(disease_path):
            continue

        count = 0

        for file in os.listdir(disease_path):

            image_path = os.path.join(disease_path, file)

            try:
                img = Image.open(image_path)

                # Verify image
                img.verify()

                # Re-open after verify()
                img = Image.open(image_path)

                if img.size != (224, 224):
                    print(f"[WARNING] Wrong size: {image_path} -> {img.size}")

                if img.mode != "RGB":
                    print(f"[WARNING] Not RGB: {image_path} -> {img.mode}")

                count += 1

            except Exception:
                corrupted.append(image_path)

        total_images += count

        print(f"{disease:<12} : {count}")

print("\n" + "=" * 60)
print(f"Total Images : {total_images}")

if corrupted:
    print("\nCorrupted Images:")
    for img in corrupted:
        print(img)
else:
    print("\n✅ No corrupted images found.")