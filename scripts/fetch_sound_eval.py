"""Build a bird-sound evaluation set from iNaturalist observations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "birdid-eval/0.1"}


def _get(url: str, timeout: int = 40) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fold(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name.replace("_", " ").lower())
    return " ".join(cleaned.split())


def load_classes(path: Path) -> dict[str, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, int] = {}
    for row in rows:
        table[_fold(row["species_name"])] = int(row["index"])
    return table


def fetch_taxon(name: str, limit: int) -> list:
    query = urllib.parse.urlencode({
        "taxon_name": name,
        "sounds": "true",
        "quality_grade": "research",
        "per_page": str(limit),
        "order_by": "votes",
    })
    payload = json.loads(_get(API + "?" + query).decode("utf-8"))
    return list(payload.get("results") or [])


def week_from_date(text: str) -> int:
    """Map an ISO date to the GeoModel 1..48 week bucket."""
    try:
        month = int(text[5:7])
        day = int(text[8:10])
    except (TypeError, ValueError):
        return 1
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return 1
    return min(48, max(1, (month - 1) * 4 + (day * 4 - 1) // 31 + 1))


def to_wav32k(source: Path, target: Path) -> bool:
    """Convert any recording to a 32 kHz mono 16-bit WAV, capped at 12 s."""
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ac", "1", "-ar", "32000", "-t", "12",
        "-c:a", "pcm_s16le", str(target),
    ]
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0 and target.exists() and target.stat().st_size > 0


def collect(name: str, key: str, observations: list, out_dir: Path, want: int) -> list:
    """Download usable recordings for one taxon, return manifest lines."""
    lines: list = []
    slug = key.replace(" ", "_")
    for observation in observations:
        if len(lines) >= want:
            break
        sounds = observation.get("observation_sounds") or []
        if not sounds:
            continue
        url = (sounds[0].get("sound") or {}).get("file_url")
        location = observation.get("location") or ""
        if not url or "," not in location:
            continue
        try:
            lat_text, lon_text = location.split(",", 1)
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            continue
        obs_id = observation.get("id")
        raw_path = out_dir / ("raw_" + slug + "_" + str(obs_id) + ".bin")
        wav_path = out_dir / ("sound_" + slug + "_" + str(obs_id) + ".wav")
        try:
            raw_path.write_bytes(_get(url, timeout=90))
        except Exception:
            raw_path.unlink(missing_ok=True)
            continue
        if not to_wav32k(raw_path, wav_path):
            raw_path.unlink(missing_ok=True)
            continue
        raw_path.unlink(missing_ok=True)
        if wav_path.stat().st_size < 64000:
            wav_path.unlink(missing_ok=True)
            continue
        week = week_from_date(observation.get("observed_on") or "")
        label = slug
        lines.append(str(wav_path) + "=" + label + "=" + f"{lat:.4f},{lon:.4f},{week}")
    del name
    return lines


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="iNaturalist 音频评测集抓取")
    parser.add_argument("classes", type=Path, help="BirdTrace classes.json")
    parser.add_argument("taxa", type=Path, help="每行一个英文俗名")
    parser.add_argument("out_dir", type=Path, help="输出目录")
    parser.add_argument("manifest", type=Path, help="输出清单 路径=标签=lat,lon,week")
    parser.add_argument("--per-taxon", type=int, default=3)
    args = parser.parse_args(argv)

    classes = load_classes(args.classes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines: list = []
    names = [line.strip() for line in args.taxa.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]

    for name in names:
        key = _fold(name)
        if key not in classes:
            print("SKIP " + name + ": not in BirdTrace", file=sys.stderr)
            continue
        try:
            observations = fetch_taxon(name, args.per_taxon * 4)
        except Exception as error:
            print("FAIL " + name + ": " + type(error).__name__, file=sys.stderr)
            continue
        if not observations:
            print("EMPTY " + name, file=sys.stderr)
            continue
        got = collect(name, key, observations, args.out_dir, args.per_taxon)
        lines.extend(got)
        print("OK " + name + " " + str(len(got)), file=sys.stderr)

    text = "\n".join(lines)
    args.manifest.write_text(text + ("\n" if lines else ""), encoding="utf-8")
    print("MANIFEST " + str(len(lines)) + " entries -> " + str(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())