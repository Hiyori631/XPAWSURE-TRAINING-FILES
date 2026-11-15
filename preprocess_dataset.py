import os
from PIL import Image

# Source dataset
SOURCE_DIR = "Datasets224new"

# Output dataset
OUTPUT_DIR = "ProcessedDataset224_PublicLocal"

# Desired image size
IMAGE_SIZE = (224, 224)

# Supported image formats
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def process_image(input_path, output_path):
    try:
        with Image.open(input_path) as img:

            # Convert grayscale/RGBA/etc. to RGB
            img = img.convert("RGB")

            # Resize
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Create output folder if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save
            img.save(output_path, quality=95)

        return True

    except Exception as e:
        print(f"❌ Failed: {input_path}")
        print(e)
        return False


processed = 0
failed = 0

for split in ["Train", "Valid", "Test"]:

    split_path = os.path.join(SOURCE_DIR, split)

    for disease in os.listdir(split_path):

        disease_path = os.path.join(split_path, disease)

        if not os.path.isdir(disease_path):
            continue

        print(f"\nProcessing {split}/{disease}")

        for file in os.listdir(disease_path):

            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            input_path = os.path.join(disease_path, file)

            output_path = os.path.join(
                OUTPUT_DIR,
                split,
                disease,
                file
            )

            if process_image(input_path, output_path):
                processed += 1
            else:
                failed += 1

print("\n===============================")
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print("===============================")
print("Dataset preprocessing complete!")