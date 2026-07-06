"""Tests for the QuantificationProfile / ProfileEntry pydantic schemas (Step 5).

Validates registry-backed metric_key validation and the JSON (de)serialization helpers
the backend's single-JSON-column storage relies on.
"""
import pytest

from iquana_toolbox.schemas.database.quantification_profile import (
    ProfileEntry,
    QuantificationProfile,
)


def test_entry_accepts_registered_metric():
    entry = ProfileEntry(metric_key="area")
    assert entry.metric_key == "area"
    assert entry.params == {}
    assert entry.label_ids is None


def test_entry_rejects_unknown_metric():
    with pytest.raises(ValueError):
        ProfileEntry(metric_key="totally_made_up")


def test_profile_rejects_unknown_metric_in_entries():
    with pytest.raises(ValueError):
        QuantificationProfile(dataset_id=1, name="bad", entries=[{"metric_key": "nope"}])


def test_entries_as_json_roundtrips():
    profile = QuantificationProfile(
        dataset_id=7,
        name="Color on cells",
        is_default=True,
        entries=[
            ProfileEntry(metric_key="mean_color_rgb", label_ids=[3, 4]),
            ProfileEntry(metric_key="area"),
        ],
    )
    json_entries = profile.entries_as_json()
    assert json_entries == [
        {"metric_key": "mean_color_rgb", "params": {}, "label_ids": [3, 4]},
        {"metric_key": "area", "params": {}, "label_ids": None},
    ]
    assert profile.metric_keys() == ["mean_color_rgb", "area"]


def test_from_db_reads_json_entries():
    class _Row:
        id = 5
        dataset_id = 7
        name = "P"
        is_default = False
        entries = [{"metric_key": "perimeter", "params": {}, "label_ids": None}]

    schema = QuantificationProfile.from_db(_Row())
    assert schema.id == 5
    assert schema.entries[0].metric_key == "perimeter"
