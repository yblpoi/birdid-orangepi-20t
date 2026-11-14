"""协议层单元测试：不依赖 NDK 或模型文件。"""

from __future__ import annotations

from birdid.protocol import (
    CODE_OK,
    CODE_PROCESSING,
    error,
    no_animals,
    normalize_class,
    ok,
    processing,
    result_entry,
    species_label,
)


def test_ok_wraps_payload_in_two_element_tuple() -> None:
    assert ok("x") == [CODE_OK, "x"]


def test_processing_carries_result_id() -> None:
    assert processing("HH-B_1") == [CODE_PROCESSING, "HH-B_1"]


def test_no_animals_message() -> None:
    assert no_animals() == [1010, "No animals"]


def test_error_keeps_code() -> None:
    assert error(1007, "quota") == [1007, "quota"]


def test_normalize_class_filters_unknown_letters() -> None:
    assert normalize_class("bmx") == "BM"
    assert normalize_class(None) == "B"
    assert normalize_class("") == "B"


def test_result_entry_rounds_box() -> None:
    entry = result_entry((1.4, 2.6, 3.2, 4.9), [[97.6, "a|b|c", 1, "B"]])
    assert entry["box"] == [1, 3, 3, 5]


def test_species_label_format() -> None:
    assert species_label("乌鸫", "Eurasian Blackbird", "Turdus merula") == "乌鸫|Eurasian Blackbird|Turdus merula"