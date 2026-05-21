import json

file_path = 'Phase_2_Plant_Classification.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update first markdown cell
nb['cells'][0]['source'][0] = '# Phase 2: Plant Species Classification - Scratch vs Pretrained Models\n'
nb['cells'][0]['source'][7] = '**Dataset**: dataset_plant_classification (Merged PlantVillage + LeafSnap field images for plant species)\n'

# Update the code cell for globbing and dataset path
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "class_dir.glob(" in line and "*.JPG" in line:
                cell['source'][i] = line.replace("class_dir.glob('*.JPG')", "list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg'))")
            elif "dataset_path = 'dataset/PlantVillage'" in line:
                cell['source'][i] = line.replace("'dataset/PlantVillage'", "'dataset_plant_classification'")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook updated successfully.')
