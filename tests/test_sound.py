"""鸟鸣链路中与分类体系无关的纯逻辑测试（不依赖真实模型文件）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from birdid.sound import (
    WEEKS_IN_YEAR,
    GeoModelFilter,
    LabelTable,
    read_wav,
    resample,
    split_windows,
    week_of_year,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_week_of_year_bounds() -> None:
    assert week_of_year(date(2024, 1, 1)) == 1
    assert week_of_year(date(2024, 12, 31)) == WEEKS_IN_YEAR
    assert 1 <= week_of_year() <= WEEKS_IN_YEAR


def test_label_table_parses_birdnet_csv(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "labels.csv",
        "idx;id;sci_name;com_name;class;order\n"
        "0;3;Abeillia abeillei;Emerald-chinned Hummingbird;Aves;Apodiformes\n"
        "1;4;Turdus merula;Eurasian Blackbird;Aves;Passeriformes\n",
    )
    table = LabelTable(path)
    assert len(table) == 2
    assert table.labels[0].species_code == "3"
    assert table.labels[0].scientific_name == "Abeillia abeillei"
    assert table.labels[1].common_name == "Eurasian Blackbird"


def test_label_table_parses_geomodel_tsv(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "geo.txt",
        "100034\tEpiaeschna heros\tSwamp Darner\n"
        "200001\tTurdus merula\tEurasian Blackbird\n",
    )
    table = LabelTable(path)
    assert len(table) == 2
    assert table.index_by_scientific_name()["turdus merula"] == 1


class _StubGeo(GeoModelFilter):
    """替身：不走 ONNX，直接给出固定的地理概率向量。"""

    def __init__(self, table: LabelTable, vector: list[float]) -> None:
        self.backend = "cpu"
        self.geo_labels = table
        self._index_by_name = table.index_by_scientific_name()
        self._session = None
        self._vector = np.asarray(vector, dtype=np.float32)

    def scores(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        return self._vector


def test_rank_uses_scientific_name_not_index(tmp_path: Path) -> None:
    table = LabelTable(
        _write(
            tmp_path / "geo.txt",
            "10\tSpecies common\tCommon\n"
            "20\tSpecies rare\tRare\n",
        )
    )
    stub = _StubGeo(table, [0.9, 0.001])
    candidates = [
        ("a", "Species rare", 0.9),
        ("b", "Species common", 0.5),
        ("c", "Unknown species", 0.8),
    ]
    ranked = stub.rank(candidates, 0.0, 0.0, 20)
    codes = [code for code, _, _ in ranked]
    # 未被 GeoModel 覆盖的物种按中性权重，不吃地理惩罚
    assert codes[0] == "c"
    assert codes[1] == "b"
    assert codes[2] == "a"
    geo = {code: geo for code, _, geo in ranked}
    assert geo["a"] == pytest.approx(0.001, rel=1e-3)
    assert geo["c"] == -1.0


def test_split_windows_pads_short_clip() -> None:
    data = np.ones(1000, dtype=np.float32)
    windows = split_windows(data, 1000, 3.0, 0.0)
    assert len(windows) == 1
    assert windows[0].shape == (3000,)
    assert windows[0][999] == 1.0 and windows[0][1000] == 0.0


def test_split_windows_steps_with_overlap() -> None:
    data = np.arange(10, dtype=np.float32)
    windows = split_windows(data, 4, 1.0, 0.5)
    assert [int(w[0]) for w in windows] == [0, 2, 4, 6]


def test_resample_changes_length() -> None:
    data = np.zeros(48000, dtype=np.float32)
    out = resample(data, 48000, 32000)
    assert out.shape[0] == 32000


def test_read_wav_roundtrip(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(np.full(4800, 16384, dtype="<i2").tobytes())
    samples, rate = read_wav(path)
    assert rate == 48000
    assert samples.shape == (4800,)
    assert samples[0] == pytest.approx(0.5, abs=1e-6)