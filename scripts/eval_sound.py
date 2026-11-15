"""Evaluate the sound path accuracy against a manifest of labelled recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from birdid.config import load_config  # noqa: E402
from birdid.sound_pipeline import SoundPipeline  # noqa: E402


def load_manifest(path: Path) -> list:
    """Parse lines of the form  <wav>=<label>=<lat>,<lon>,<week>."""
    items: list = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit("=", 2)
        if len(parts) != 3:
            continue
        wav_text, label, geo_text = parts
        try:
            lat_text, lon_text, week_text = geo_text.split(",")
            geo = (float(lat_text), float(lon_text), int(week_text))
        except ValueError:
            geo = (None, None, None)
        items.append((Path(wav_text), label, geo))
    return items


ALIASES = {
    "common chaffinch": "eurasian chaffinch",
    "eurasian blackbird": "common blackbird",
}


def canonical(name: str) -> str:
    """Normalise a common name and fold known synonym pairs."""
    key = " ".join(name.replace("_", " ").strip().lower().split())
    return ALIASES.get(key, key)


def top_labels(entries, count: int) -> list:
    """Flatten the protocol entries into a list of common names."""
    names: list = []
    for entry in entries or []:
        for row in entry.get("list") or []:
            names.append(str(row[1]).split("|")[0].strip().replace(" ", "_"))
    return names[:count]


def evaluate(pipeline: SoundPipeline, items: list, use_geo: bool) -> tuple:
    """Return (hits, total, per-item records)."""
    hits = 0
    total = 0
    records: list = []
    for wav_path, label, geo in items:
        if not wav_path.exists():
            continue
        raw = wav_path.read_bytes()
        if use_geo and geo[0] is not None:
            entries = pipeline.predict(raw, geo[0], geo[1], geo[2])
        else:
            entries = pipeline.predict(raw)
        total += 1
        names = top_labels(entries, 5)
        expected = canonical(label)
        lowered = [canonical(name) for name in names]
        hit = bool(lowered) and lowered[0] == expected
        in_top5 = expected in lowered
        hits += int(hit)
        records.append((wav_path.name, label, names[0] if names else "", hit, in_top5))
    return hits, total, records


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="鸟鸣链路精度评测")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    items = load_manifest(args.manifest)
    print("MANIFEST " + str(len(items)) + " entries")
    pipeline = SoundPipeline(config)
    try:
        base_hits, base_total, _ = evaluate(pipeline, items, use_geo=False)
        geo_hits, geo_total, records = evaluate(pipeline, items, use_geo=True)
    finally:
        pipeline.close()

    def pct(hit: int, total: int) -> str:
        return "0.0" if total == 0 else f"{100.0 * hit / total:.1f}"

    print(f"BASE {base_hits}/{base_total} {pct(base_hits, base_total)}%")
    print(f"GEO  {geo_hits}/{geo_total} {pct(geo_hits, geo_total)}%")
    top5 = sum(1 for rec in records if rec[4])
    print(f"TOPK {top5}/{geo_total} {pct(top5, geo_total)}%")
    if args.verbose:
        for name, expected, got, hit, in_top5 in records:
            flag = "HIT " if hit else ("T5  " if in_top5 else "MISS")
            print(f"{flag} {name} expected={expected} got={got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())