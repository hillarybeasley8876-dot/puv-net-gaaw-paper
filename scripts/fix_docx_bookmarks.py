#!/usr/bin/env python3
"""Renumber docx-js bookmark IDs while preserving bookmark names and fields."""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fix_docx_bookmarks.py file.docx")
    target = Path(sys.argv[1]).resolve()
    if not target.exists() or target.suffix.lower() != ".docx":
        raise SystemExit(f"not a docx: {target}")

    with zipfile.ZipFile(target, "r") as zin:
        xml = zin.read("word/document.xml")
        root = ET.fromstring(xml)
        stack: list[str] = []
        next_id = 1
        n_start = n_end = 0
        for elem in root.iter():
            if elem.tag == W + "bookmarkStart":
                new_id = str(next_id)
                next_id += 1
                elem.set(W + "id", new_id)
                stack.append(new_id)
                n_start += 1
            elif elem.tag == W + "bookmarkEnd":
                if not stack:
                    raise RuntimeError("bookmarkEnd without bookmarkStart")
                elem.set(W + "id", stack.pop())
                n_end += 1
        if stack:
            raise RuntimeError(f"unclosed bookmarks: {stack}")
        # ElementTree drops namespace declarations that are not used by an
        # element but would otherwise leave stale prefixes in mc:Ignorable.
        root.attrib.pop(MC + "Ignorable", None)
        fixed = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        entries = [(info, fixed if info.filename == "word/document.xml" else zin.read(info.filename))
                   for info in zin.infolist()]

    fd, tmp_name = tempfile.mkstemp(prefix="bookmark_fix_", suffix=".docx", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info, data in entries:
                zout.writestr(info, data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Renumbered {n_start} bookmark pairs in {target}")


if __name__ == "__main__":
    main()
