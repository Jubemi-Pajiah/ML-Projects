import os
from pathlib import Path
from PIL import Image

input_dir = Path("rock_dataset")        
output_dir = Path("rock_dataset_clean") 
target_size = (224, 224)           


def normalize_dataset(input_dir, output_dir, target_size=(224,224)):
    class_names = []

    for cls in os.listdir(input_dir):
        cls_path = input_dir / cls
        if not cls_path.is_dir():
            continue

        # Capitalizing the folder names
        cls_name = cls.capitalize()
        class_names.append(cls_name)

        out_cls_path = output_dir / cls_name
        out_cls_path.mkdir(parents=True, exist_ok=True)

        counter = 1
        for file in os.listdir(cls_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = cls_path / file
                    img = Image.open(img_path).convert("RGB")


                    img = img.resize(target_size)

                    out_name = f"{counter}.jpg"
                    out_path = out_cls_path / out_name
                    img.save(out_path, "JPEG", quality=95)

                    print(f"{img_path} ++++ {out_path}")
                    counter += 1
                except Exception as e:
                    print(f"Error with {file}: {e}")

    print("\nNormalization complete!")
    print("Classes found:")
    for name in class_names:
        print(f"- {name}")

if __name__ == "__main__":
    normalize_dataset(input_dir, output_dir, target_size)
