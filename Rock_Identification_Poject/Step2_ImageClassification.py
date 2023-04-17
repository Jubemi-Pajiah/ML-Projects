import os, shutil, random
from pathlib import Path
import pandas as pd


input_dir = Path("Rock_Identification_Poject/Outputs/rock_dataset_clean")  
output_dir = Path("Rock_Identification_Poject/Outputs/rock_dataset_split") 
ratios = (0.7, 0.15, 0.15)   
random.seed(42)          


def split_dataset(input_dir, output_dir, ratios=(0.7,0.15,0.15)):
    summary = []

    for cls in os.listdir(input_dir):
        cls_path = input_dir / cls
        if not cls_path.is_dir():
            continue

        imgs = list(cls_path.glob("*.jpg"))
        random.shuffle(imgs)
        n = len(imgs)

        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train+n_val]
        test_imgs = imgs[n_train+n_val:]

        for split_name, split_list in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            out_dir = output_dir / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in split_list:
                shutil.copy(p, out_dir / p.name)

        summary.append({
            "Class": cls,
            "Total": n,
            "Train": len(train_imgs),
            "Val": len(val_imgs),
            "Test": len(test_imgs)
        })

    df = pd.DataFrame(summary).sort_values("Total", ascending=False)

    # Saving the summary in a CSV file
    csv_path = output_dir / "class_distribution.csv"
    df.to_csv(csv_path, index=False)

    print("\nSplit complete!")
    print(f"Distribution saved to {csv_path}")
    print(df)

    return df

if __name__ == "__main__":
    split_dataset(input_dir, output_dir, ratios)
