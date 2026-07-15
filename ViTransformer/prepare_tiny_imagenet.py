"""Download and prepare tiny-imagenet-200 into the ImageFolder layout that
engine/data.py expects.

    python prepare_tiny_imagenet.py                    # -> ./../datasets/tiny-imagenet-200
    python prepare_tiny_imagenet.py --root /data       # custom datasets root

Idempotent: re-running skips steps that are already done. After it finishes:

    <root>/tiny-imagenet-200/train/<wnid>/images/*.JPEG   (ImageFolder-ready as-is)
    <root>/tiny-imagenet-200/val/<wnid>/*.JPEG            (rebuilt from val_annotations.txt)

Then:  python train.py --config configs/tiny-imagenet-baseline.yaml
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Canonical Stanford CS231n mirror (~240 MB). Note: plain http.
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "tiny-imagenet-200.zip"
    if zip_path.exists():
        print(f"zip already present: {zip_path}")
        return zip_path

    print(f"downloading {URL}\n  -> {zip_path}  (~240 MB)")

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            print(f"\r  {done >> 20} / {total_size >> 20} MiB "
                  f"({100 * done / total_size:5.1f}%)", end="")

    urllib.request.urlretrieve(URL, zip_path, _progress)
    print()
    return zip_path


def extract(zip_path: Path, root: Path) -> Path:
    dataset_dir = root / "tiny-imagenet-200"
    if dataset_dir.exists():
        print(f"already extracted: {dataset_dir}")
        return dataset_dir
    print(f"extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return dataset_dir


def reorganize_val(dataset_dir: Path) -> None:
    """Turn the flat val/images/ split into ImageFolder per-class subdirs."""
    val = dataset_dir / "val"
    annotations = val / "val_annotations.txt"
    images = val / "images"

    if not annotations.exists():
        print(f"val already reorganized (no {annotations.name}); skipping")
        return

    print("reorganizing val/ into per-class subfolders ...")
    # Each line: <filename>\t<wnid>\t<x>\t<y>\t<w>\t<h>
    mapping = {}
    for line in annotations.read_text().splitlines():
        if not line.strip():
            continue
        fname, wnid = line.split("\t")[:2]
        mapping[fname] = wnid

    for fname, wnid in mapping.items():
        cls_dir = val / wnid
        cls_dir.mkdir(exist_ok=True)
        src = images / fname
        if src.exists():
            shutil.move(str(src), str(cls_dir / fname))

    shutil.rmtree(images, ignore_errors=True)   # now-empty flat dir
    annotations.unlink(missing_ok=True)
    print(f"  val reorganized into {len(set(mapping.values()))} class folders")


def main():
    ap = argparse.ArgumentParser(description="Fetch & prepare tiny-imagenet-200.")
    ap.add_argument("--root", default="./../datasets",
                    help="Datasets root (must match data.root in the config).")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the downloaded zip instead of deleting it.")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    zip_path = download(root)
    dataset_dir = extract(zip_path, root)
    reorganize_val(dataset_dir)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"removed {zip_path.name}")

    print(f"\nReady: {dataset_dir}")
    print("Train with: python train.py --config configs/tiny-imagenet-baseline.yaml")


if __name__ == "__main__":
    main()
