#!/usr/bin/env python3
"""Prepare schematic figures for Word by removing embedded title bands.

The formal caption is provided below each image in the thesis, so duplicated
titles such as "图 2-3 ..." inside the bitmap are cropped from the Word copy.
Source figures remain unchanged.
"""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "figures_schematic"
OUT = ROOT / "docs" / "figures_word"
NAMES = [
    "F1_1_task_illustration.png",
    "F1_2_technical_roadmap.png",
    "F1_3_chapter_map.png",
    "F2_1_method_timeline.png",
    "F2_2_method_taxonomy.png",
    "F2_3_backbone_evolution.png",
    "F2_4_loss_quality_map.png",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        source = SRC / name
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as img:
            crop_top = min(70, max(0, img.height // 10))
            result = img.crop((0, crop_top, img.width, img.height))
            result.save(OUT / name, optimize=True)
    print(f"Prepared {len(NAMES)} Word figures in {OUT}")


if __name__ == "__main__":
    main()
