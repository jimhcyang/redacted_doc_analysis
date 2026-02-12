from __future__ import annotations

import argparse
from pathlib import Path


def _is_sidecar(path: Path) -> bool:
    name = path.name
    if "Zone.Identifier" in name:
        return True
    if name.startswith("._"):
        return True
    return False


def _safe_str(path: Path) -> str:
    return str(path).encode("cp1252", errors="replace").decode("cp1252")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Root folder to scan")
    parser.add_argument("--apply", action="store_true", help="Actually delete files/folders")
    args = parser.parse_args()

    root = Path(args.root)
    files = [p for p in root.rglob("*") if p.is_file() and _is_sidecar(p)]
    macos_dirs = [p for p in root.rglob("__MACOSX") if p.is_dir()]

    print(f"[INFO] sidecar files: {len(files)}")
    print(f"[INFO] __MACOSX dirs: {len(macos_dirs)}")
    for p in files[:20]:
        print(f"  file: {_safe_str(p)}")
    for p in macos_dirs[:20]:
        print(f"  dir : {_safe_str(p)}")

    if not args.apply:
        print("[INFO] dry-run mode; rerun with --apply to delete")
        return

    deleted_files = 0
    for p in files:
        try:
            p.unlink(missing_ok=True)
            deleted_files += 1
        except Exception as e:
            print(f"[WARN] failed to delete file {_safe_str(p)}: {e}")

    deleted_dirs = 0
    for d in macos_dirs:
        try:
            for child in sorted(d.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            d.rmdir()
            deleted_dirs += 1
        except Exception as e:
            print(f"[WARN] failed to delete dir {_safe_str(d)}: {e}")

    print(f"[INFO] deleted files: {deleted_files}")
    print(f"[INFO] deleted dirs : {deleted_dirs}")


if __name__ == "__main__":
    main()
