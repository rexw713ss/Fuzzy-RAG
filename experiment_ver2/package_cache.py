"""Package only the small cache files needed by ver2; never model/index data.

Run in the original experimenter's checkout:
python -m experiment_ver2.package_cache --root D:/Han/rex_rag/data/full --out first_run_cache.zip
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from experiment_ver2.data import discover_datasets, load_dataset


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    if root in output.parents:
        raise ValueError("write the archive outside the original cache root")
    datasets = discover_datasets(root)
    for name in datasets:
        load_dataset(root, name)
    paths = [root / "splits.json"]
    for name in datasets:
        paths.extend(root / name / f"{stem}.jsonl" for stem in ("queries", "signals", "outcomes"))
        optional = [root / name / "clusters.jsonl", root / "results" / f"{name}.json"]
        paths.extend(p for p in optional if p.is_file())
    manifest = {"schema": 1, "purpose": "first-run cache for exploratory ver2 replay",
                "datasets": datasets,
                "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("CACHE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = package(args.root, args.out)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot package cache: {exc}\n")
    print(f"Validated and packaged {len(manifest['files'])} files: {args.out}")


if __name__ == "__main__":
    main()
