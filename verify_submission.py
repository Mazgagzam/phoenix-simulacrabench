"""Verify or rebuild the Phoenix payload from the submitted file manifest."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
TABICL = "weights/tabicl-classifier-v2-20260212.ckpt"
TABPFN = "weights/tabpfn-v3.5-fast-20260909.safetensors"
EMBEDDING = "vendor/tabpfn/architectures/shared/tabpfn_col_embedding.pt"


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path, manifest, exact=False):
    with path.open("rb") as stream:
        archive_hash = digest(stream)
    if exact and archive_hash != manifest["archive_sha256"]:
        raise ValueError("Archive SHA-256 differs from the submitted ZIP")
    with ZipFile(path) as archive:
        entries = archive.namelist()
        if len(entries) != len(manifest["members"]) or set(entries) != set(manifest["members"]):
            raise ValueError("Archive member list differs")
        for name, expected in manifest["members"].items():
            with archive.open(name) as stream:
                if digest(stream) != expected:
                    raise ValueError(f"File SHA-256 differs: {name}")
    print(f"Verified {len(entries)} files. ZIP SHA-256: {archive_hash}")


def rebuild(output, manifest, tabicl, tabpfn, wheel_path):
    assets = {TABICL: tabicl, TABPFN: tabpfn}
    with wheel_path.open("rb") as stream:
        if digest(stream) != manifest["tabpfn_wheel_sha256"]:
            raise ValueError("TabPFN wheel SHA-256 differs")
    with ZipFile(wheel_path) as wheel:
        embedding = wheel.read(EMBEDDING.removeprefix("vendor/"))
    for name, expected in manifest["members"].items():
        if name == EMBEDDING:
            result = hashlib.sha256(embedding).hexdigest()
        else:
            with assets.get(name, ROOT / "submission" / name).open("rb") as stream:
                result = digest(stream)
        if result != expected:
            raise ValueError(f"Source or asset SHA-256 differs: {name}")

    with ZipFile(output, "x") as archive:
        for name in sorted(manifest["members"]):
            entry = ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            entry.create_system = 3
            entry.compress_type = ZIP_DEFLATED
            if name == EMBEDDING:
                archive.writestr(entry, embedding)
                continue
            with assets.get(name, ROOT / "submission" / name).open("rb") as source:
                with archive.open(entry, "w") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    verify_archive(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", type=Path)
    mode.add_argument("--output", type=Path)
    parser.add_argument("--tabicl", type=Path)
    parser.add_argument("--tabpfn", type=Path)
    parser.add_argument("--tabpfn-wheel", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "submission_manifest.json").read_text(encoding="utf-8"))
    if args.archive:
        verify_archive(args.archive, manifest, exact=True)
    else:
        if not all((args.tabicl, args.tabpfn, args.tabpfn_wheel)):
            parser.error("--output requires --tabicl, --tabpfn and --tabpfn-wheel")
        rebuild(args.output, manifest, args.tabicl, args.tabpfn, args.tabpfn_wheel)


if __name__ == "__main__":
    main()
