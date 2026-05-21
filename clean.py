import os
import shutil

dataset_root = "dataset/PlantVillage"

for root, dirs, files in os.walk(dataset_root):
    for file in files:
        cleaned = file.rstrip().replace("\xa0", "")
        
        if file != cleaned:
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, cleaned)
            
            print(f"Renaming: '{file}' → '{cleaned}'")
            shutil.move(old_path, new_path)

print("✅ Done!")