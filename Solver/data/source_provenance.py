"""Resolve original downloaded pages, including legacy MangaZero puzzles."""

from functools import lru_cache
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def portable_path(value):
    return Path(str(value).replace("\\", "/"))


@lru_cache(maxsize=16)
def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@lru_cache(maxsize=8)
def raw_manifest_index(root):
    path = root / "manifest.jsonl"
    if not path.is_file():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("image_path"):
                key = str(entry["image_path"]).replace("\\", "/")
                entries[key] = entry
    return entries


def raw_source_location(panel, source_dir, raw_dir=None):
    """Return (raw sample directory or original file, recorded source ID).

    MangaZero source_image_path is an upstream page ID. Resolve it through the
    raw manifest to the local numbered sample directory, without picking sides.
    """
    source_dir = Path(source_dir).resolve()
    recorded = panel.get("source_image_path") or panel.get("source_path") or ""
    if not recorded:
        return None, ""
    key = str(recorded).replace("\\", "/")
    candidates = []
    if raw_dir is not None:
        candidates.append(Path(raw_dir).resolve())
    else:
        configured = read_json(source_dir / "meta.json").get("raw_dir")
        if configured:
            path = portable_path(configured)
            candidates.extend([path.resolve(), ROOT / path, source_dir / path, source_dir.parent / path])
        candidates.extend([source_dir.parent / "raw", ROOT / "Mangazero/raw", ROOT / "Data/Mangazero/raw"])
    for root in dict.fromkeys(path.resolve() for path in candidates):
        entry = raw_manifest_index(root).get(key)
        if not entry:
            continue
        for field in ("annotation_path", "url1_path", "url2_path"):
            if entry.get(field):
                return (root / portable_path(entry[field])).resolve().parent, key
    # Other datasets may already store an original file path directly.
    path = portable_path(recorded)
    candidates = [path] if path.is_absolute() else [source_dir / path, Path.cwd() / path, source_dir.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), key
    return None, key
