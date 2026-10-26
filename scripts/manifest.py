#!/usr/bin/env python3
"""Emit a pipe-separated download manifest for pinned Hugging Face revisions.

Usage: manifest.py <repo> <revision> [path ...]

Columns: path, size in bytes, LFS SHA-256 (empty when not LFS), revision.
With no path filter, every .json/.jinja file is listed. Standard library only.
"""

import json
import sys
import urllib.request

DEFAULT_SUFFIXES = (".json", ".jinja")


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: manifest.py <repo> <revision> [path ...]")
    repo, revision, *wanted = sys.argv[1:]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.load(response)
    for row in sorted(rows, key=lambda item: item["path"]):
        if row.get("type") != "file":
            continue
        path = row["path"]
        if wanted:
            if path not in wanted:
                continue
        elif not path.endswith(DEFAULT_SUFFIXES):
            continue
        size = row.get("size")
        if not isinstance(size, int):
            raise SystemExit(f"{path}: manifest has no size")
        oid = (row.get("lfs") or {}).get("oid", "")
        print(f"{repo}|{path}|{size}|{oid}|{revision}")


if __name__ == "__main__":
    main()
