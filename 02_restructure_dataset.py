import shutil
from pathlib import Path


SRC_ROOT_PV = Path("dataset_clean/dataset/PlantVillage")
SRC_ROOT_LS_FIELD = Path("dataset_clean/dataset/leafsnap-dataset/dataset/images/field")
SRC_ROOT_LS_LAB = Path("dataset_clean/dataset/leafsnap-dataset/dataset/images/lab")
DST_ROOT = Path("dataset_plant_classification")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def reset_destination(dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)


def iter_image_files(folder: Path):
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield file_path


def normalize_plantvillage_label(raw_name: str) -> str:
    if "___" in raw_name:
        plant_name = raw_name.split("___", 1)[0]
    else:
        plant_name = raw_name.split("_", 1)[0]

    plant_name = plant_name.replace("__bell", "")
    return plant_name.strip().capitalize()


def normalize_leafsnap_label(raw_name: str) -> str:
    # Leafsnap folder names use scientific names like acer_rubrum.
    # We group by genus to keep labels broad instead of species-specific.
    genus = raw_name.split("_", 1)[0].strip()
    return genus.capitalize()


def copy_images(source_dir: Path, target_dir: Path, prefix: str) -> int:
    copied = 0
    target_dir.mkdir(parents=True, exist_ok=True)

    for img in iter_image_files(source_dir):
        dst_file = target_dir / f"{prefix}_{source_dir.name}_{img.name}"
        if not dst_file.exists():
            shutil.copy2(img, dst_file)
            copied += 1

    return copied


def process_plantvillage(src_root: Path, dst_root: Path) -> dict:
    summary = {}
    print("Processing PlantVillage...")

    if not src_root.exists():
        print(f"Warning: {src_root} not found.")
        return summary

    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue

        unified_name = normalize_plantvillage_label(class_dir.name)
        target_dir = dst_root / unified_name
        copied = copy_images(class_dir, target_dir, "pv")
        summary[unified_name] = summary.get(unified_name, 0) + copied

    return summary


def process_leafsnap(src_root: Path, dst_root: Path, source_tag: str) -> dict:
    summary = {}
    print(f"Processing Leafsnap {source_tag}...")

    if not src_root.exists():
        print(f"Warning: {src_root} not found.")
        return summary

    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue

        unified_name = normalize_leafsnap_label(class_dir.name)
        target_dir = dst_root / unified_name
        copied = copy_images(class_dir, target_dir, f"ls_{source_tag}")
        summary[unified_name] = summary.get(unified_name, 0) + copied

    return summary


def merge_counts(*summaries: dict) -> dict:
    merged = {}
    for summary in summaries:
        for label, count in summary.items():
            merged[label] = merged.get(label, 0) + count
    return merged


def main() -> None:
    print("Starting to restructure dataset...")
    reset_destination(DST_ROOT)

    pv_summary = process_plantvillage(SRC_ROOT_PV, DST_ROOT)
    ls_field_summary = process_leafsnap(SRC_ROOT_LS_FIELD, DST_ROOT, "field")
    ls_lab_summary = process_leafsnap(SRC_ROOT_LS_LAB, DST_ROOT, "lab")

    final_summary = merge_counts(pv_summary, ls_field_summary, ls_lab_summary)

    print("\nFinal class counts:")
    for label in sorted(final_summary):
        print(f"- {label}: {final_summary[label]}")

    print(f"\nTotal classes: {len(final_summary)}")
    print(f"Output folder: {DST_ROOT}")
    print("Done.")


if __name__ == "__main__":
    main()
