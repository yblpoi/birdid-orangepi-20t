"""Build a bird-photo evaluation set from iNaturalist observations."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "birdid-eval/0.1"}


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path) -> bool:
    """Fetch one photo and verify it really decodes as an image."""
    try:
        request = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(request, timeout=60) as response:
            dest.write_bytes(response.read())
        Image.open(dest).load()
        return True
    except Exception as error:
        print("  download failed: " + str(error), file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False


def pick(observation: dict, known: set):
    """Return (label, photo_url) when the observation maps onto a known class."""
    taxon = observation.get("taxon") or {}
    common = (taxon.get("preferred_common_name") or "").strip()
    if not common:
        return None
    label = common.replace(" ", "_")
    if label not in known:
        return None
    photos = observation.get("photos") or []
    if not photos:
        return None
    return label, photos[0]["url"].replace("/square.", "/large.")


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="iNaturalist 图片评测集抓取")
    parser.add_argument("classes", type=Path, help="BirdTrace classes.json")
    parser.add_argument("taxa", type=Path, help="每行一个拉丁名或英文俗名")
    parser.add_argument("out_dir", type=Path, help="图片输出目录")
    parser.add_argument("manifest", type=Path, help="输出清单 路径=标签")
    parser.add_argument("--per-taxon", type=int, default=1)
    args = parser.parse_args(argv)

    known = {row["species_name"] for row in json.loads(args.classes.read_text(encoding="utf-8"))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines: list = []
    names = [line.strip() for line in args.taxa.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]

    for taxon in names:
        query = urllib.parse.urlencode({
            "taxon_name": taxon,
            "photos": "true",
            "quality_grade": "research",
            "per_page": "10",
        })
        try:
            payload = _get(API + "?" + query)
        except Exception as error:
            print("SKIP " + taxon + " (" + type(error).__name__ + ")", file=sys.stderr)
            continue
        taken = 0
        for observation in payload.get("results") or []:
            if taken >= args.per_taxon:
                break
            result = pick(observation, known)
            if result is None:
                continue
            label, url = result
            dest = args.out_dir / ("eval_" + label + "_" + str(observation.get("id")) + ".jpg")
            if not dest.exists() and not _download(url, dest):
                continue
            lines.append(str(dest) + "=" + label)
            taken += 1
        print("OK " + taxon + " " + str(taken), file=sys.stderr)
        time.sleep(0.3)

    args.manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print("MANIFEST " + str(len(lines)) + " entries -> " + str(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())