import argparse
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid", "both"], default="train")
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid"] if args.split == "both" else [args.split]
    for split in splits:
        url = URLS[split]
        archive = root / f"DIV2K_{split}_HR.zip"
        out_dir = root / f"DIV2K_{split}_HR"
        if out_dir.exists() and any(out_dir.rglob("*.png")):
            print(f"{out_dir} already contains images; skip.")
            continue
        if not archive.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
        print(f"Extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
