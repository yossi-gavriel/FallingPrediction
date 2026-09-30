"""Delete generated demo/verification outputs, and nothing else. Standard library only.

Usage: python scripts/handoff/clean_outputs.py [--dry-run]

Removes only the sub-folders of demo_outputs/ that contain the marker file ".falls_ml_generated" (written by the
handoff scripts), then demo_outputs/ itself if it is marked and nothing unmarked is left in it. Unmarked folders,
loose files and links are always kept and listed. data/, configs/, src/, runs/ and reports/ are never touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def plan_cleanup(demo_root: Path) -> tuple[list[Path], list[Path], bool]:
    """Return (folders to delete, entries kept, whether demo_root itself is deleted afterwards)."""
    remove: list[Path] = []
    keep: list[Path] = []
    if not demo_root.exists():
        return remove, keep, False
    if demo_root.is_symlink() or not demo_root.is_dir():
        return remove, [demo_root], False
    own_files = {common.MARKER_FILE, common.SYNTHETIC_NOTICE_FILE}
    for entry in sorted(demo_root.iterdir()):
        if entry.name in own_files and entry.is_file() and not entry.is_symlink():
            continue
        if common.is_generated(entry):
            remove.append(entry)
        else:
            keep.append(entry)
    return remove, keep, common.is_generated(demo_root) and not keep


def clean(demo_root: Path, *, dry_run: bool) -> int:
    remove, keep, remove_root = plan_cleanup(demo_root)
    verb = "would remove" if dry_run else "removed"
    if not demo_root.exists():
        print(f"Nothing to clean: {demo_root} does not exist.")
        return 0
    for entry in remove:
        if not dry_run:
            common.remove_tree(entry)
        print(f"{verb}: {entry}")
    for entry in keep:
        print(f"kept (not created by these scripts, no {common.MARKER_FILE}): {entry}")
    if remove_root:
        if not dry_run:
            common.remove_tree(demo_root)
        print(f"{verb}: {demo_root}")
    if not remove and not remove_root:
        print("Nothing to remove.")
    print(f"{'Dry run: nothing was deleted.' if dry_run else 'Done.'} {len(remove) + int(remove_root)} folder(s) "
          f"{'would be removed' if dry_run else 'removed'}, {len(keep)} kept.")
    return 0


def main(argv: list[str] | None = None) -> int:
    common.configure_console()
    ap = argparse.ArgumentParser(description="Delete marker-protected demo outputs under demo_outputs/.")
    ap.add_argument("--dry-run", action="store_true", help="list what would be removed without deleting anything")
    ap.add_argument("--root", default=str(common.PROJECT_ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        return clean(Path(args.root) / common.DEMO_OUTPUTS, dry_run=args.dry_run)
    except OSError as exc:
        print(f"ERROR: cleaning failed: {exc}\nClose programs that use files in demo_outputs (Excel, editors) and try again.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
