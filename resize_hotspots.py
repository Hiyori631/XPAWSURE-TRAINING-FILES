from PIL import Image
import os

input_folder = "cropped_datasets/int_train_hotspot_2"
output_folder = "cropped_datasets/int_train_hotspot_2 - 224"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)

        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            img = img.resize((224, 224))

            save_path = os.path.join(output_folder, filename)
            img.save(save_path)

            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Error with {filename}: {e}")

print("Hot Spots images resized successfully!")