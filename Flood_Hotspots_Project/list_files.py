# Flood_Hotspots_Project/list_files.py
import os
from Flood_Hotspots_Project.config import base_dir

def list_all_files(directory):
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    print(f"Listing all files under: {base_dir}\n")
    list_all_files(base_dir)
