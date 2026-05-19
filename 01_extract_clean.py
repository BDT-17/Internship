import zipfile
import os

zip_path = "dataset.zip"
extract_path = "dataset_clean"

print(f"Starting extraction from {zip_path} to {extract_path}...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    total_files = len(zip_ref.infolist())
    print(f"Total items: {total_files}")
    
    for i, member in enumerate(zip_ref.infolist()):
        clean_name = member.filename.rstrip()  # Xóa trailing space
        
        # skip directory
        if member.is_dir():
            continue
        
        source = zip_ref.open(member)
        target_path = os.path.join(extract_path, clean_name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, "wb") as target:
            target.write(source.read())
            
        if (i + 1) % 5000 == 0:
            print(f"Extracted {i+1}/{total_files}...")

print("✓ Extracted with cleaned filenames")
