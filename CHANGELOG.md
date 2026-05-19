# Changelog

All notable changes to the **Plant Species Classification Experiment** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] - 2026-05-19

### Added
- `PROJECT_XRAY.md` serving as comprehensive codebase documentation generated via the Vibecode Kit v5.0 X-Ray Protocol.
- `CHANGELOG.md` to track project revision history.

### Changed
- Improved dataset description in `README.md` to reflect classification labels consolidation (crop-level for PlantVillage and genus-level for LeafSnap).
- Configured helper utilities like `update_nb.py` and `fix_notebook_path.py` to prevent local absolute paths from leaking into notebook settings.

---

## [1.0.0] - 2026-05-15

### Added
- Primary implementation Jupyter notebook `Phase_2_Plant_Classification.ipynb` evaluating:
  - Custom CNN trained from scratch.
  - ResNet50 Feature Extraction (frozen backbone).
  - ResNet50 Fine-Tuning (progressive layers unfreezing).
- Raw data extraction helper script `01_extract_clean.py` addressing trailing spaces issues on Windows.
- Dataset consolidation script `02_restructure_dataset.py` implementing crop/genus folder clustering.
- Data verification check script `clean.py` to strip invalid non-breaking space characters.
- Programmatic visualization helper `add_visualization_cells.py` to automatically insert metrics plots.
- Documentation including `IMPLEMENTATION_SUMMARY.md` and literature research foundation `Phase_1_Literature_Review_Report.md`.
- `requirements.txt` containing dependencies pinning PyTorch, Torchvision, Scikit-Learn, and Jupyter.
