# Kaggle Git Workflow

This repo is now organized so Kaggle notebooks can stay thin and always run the latest Python code from Git.

## Full training notebook cell

Enable Internet in Kaggle, then put this in the first notebook cell:

```python
REPO_URL = "https://github.com/<your-user>/<your-repo>.git"
REPO_DIR = "/kaggle/working/Internship"

!python - <<'PY'
import subprocess
from pathlib import Path

repo_url = "https://github.com/<your-user>/<your-repo>.git"
repo_dir = Path("/kaggle/working/Internship")

if (repo_dir / ".git").exists():
    subprocess.run(["git", "fetch", "--all"], cwd=repo_dir, check=True)
    subprocess.run(["git", "pull", "--ff-only"], cwd=repo_dir, check=True)
else:
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
PY

%cd /kaggle/working/Internship
!python scripts/train_kaggle.py --epochs 50 --batch-size 32
```

## Faster smoke run

```python
%cd /kaggle/working/Internship
!python scripts/train_kaggle.py --epochs 1 --batch-size 16 --models custom_cnn --no-save-every-epoch
```

## Showcase inference

Attach the dataset and a previous `plant_training_outputs` artifact to the Kaggle notebook, then run:

```python
%cd /kaggle/working/Internship
!python scripts/infer_kaggle.py --model-name resnet50_fine_tuning --top-k 5
```

## Outputs

Training writes to `/kaggle/working/plant_training_outputs` by default:

- `class_names.json`
- `class_counts.csv`
- per-model `best_model.pth`
- per-model `history.csv`
- per-model plots and confusion matrices
- `all_models_summary.csv`
- `model_comparison.png`

