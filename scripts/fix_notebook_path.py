"""
Fix hardcoded path in notebook cell 4
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "Phase_2_Plant_Classification.ipynb"

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix cell 4 (the data loading cell)
for cell in notebook['cells']:
    if 'def load_plantvillage_dataset' in ''.join(cell.get('source', [])):
        # Replace the hardcoded path line
        source = cell['source']
        for i, line in enumerate(source):
            if 'dataset_path = Path("D:\\DS\\internship\\dataset\\PlantVillage")' in line:
                source[i] = '    dataset_path = Path(dataset_path)\n'
                print(f"Fixed line {i}: Changed hardcoded path to use parameter")
                break

# Write back
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Notebook path fixed successfully!")
