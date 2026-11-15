"""构建 BirdTrace 类名 → GeoModel 分类的映射（按英文俗名匹配）。

BirdTrace 的 classes.json 只有英文俗名，没有学名；GeoModel 的标签表同时有
学名和英文俗名，因此按「归一化俗名」把两边对齐，既能补出学名，也能拿到
GeoModel 的行号用于地理概率查询。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def normalize(name: str) -> str:
    """归一化俗名：小写、去撇号、下划线转空格、压缩空白。

    两边表的差异主要在撇号：BirdTrace 写 Abbotts_Babbler，GeoModel 写
    "Abbott's Babbler"。先去掉撇号再比，匹配率从 81.9% 提到 88.3%。
    """
    text = name.replace("_", " ").strip().lower()
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_geo_labels(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def main() -> None:
    classes_path = Path(sys.argv[1])
    geo_labels_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    geo = load_geo_labels(geo_labels_path)

    by_common: dict[str, int] = {}
    by_sci: dict[str, int] = {}
    for idx, (_code, sci, common) in enumerate(geo):
        by_common.setdefault(normalize(common), idx)
        by_sci.setdefault(normalize(sci), idx)

    mapping: dict[str, list] = {}
    sci_by_index: dict[str, str] = {}
    matched = 0
    for entry in classes:
        index = int(entry["index"])
        name = entry["species_name"]
        geo_idx = by_common.get(normalize(name))
        if geo_idx is None:
            continue
        matched += 1
        sci = geo[geo_idx][1]
        mapping[str(index)] = [geo_idx, sci]
        sci_by_index[str(index)] = sci

    payload = {"species_scientific": sci_by_index, "geomodel_index": {k: v[0] for k, v in mapping.items()}}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    total = len(classes)
    print("classes=%d matched=%d (%.1f%%) -> %s" % (total, matched, 100.0 * matched / total, out_path))


if __name__ == "__main__":
    main()
