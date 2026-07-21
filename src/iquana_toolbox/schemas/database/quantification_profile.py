"""Pydantic schemas for quantification profiles.

A *quantification profile* is a named, per-dataset selection of which metrics to
compute / report and, per metric, an optional parameter dict and an optional label
scoping. It is the user-facing flexibility layer (Step 5): different use-cases want
different metrics on different labels, instead of the hard-coded four geometry metrics.

Storage: the backend keeps the whole profile as one row with the entry list serialized
into a single JSON column (see ``app.database.quantification_profiles``), so these
schemas double as the JSON (de)serialization contract via :meth:`QuantificationProfile.from_db`
and :meth:`QuantificationProfile.entries_as_json`.
"""
from logging import getLogger

from pydantic import BaseModel, Field, field_validator

from iquana_toolbox.quantification import METRIC_REGISTRY

logger = getLogger(__name__)


class ProfileEntry(BaseModel):
    """A single metric selection within a :class:`QuantificationProfile`.

    ``metric_key`` must be a key registered in the metric registry. ``params`` is a
    forward-looking per-metric parameter dict (e.g. a future per-profile ``k`` for the
    knn metric); it is validated to be a dict but not yet acted on by the compute path
    (see the module docstring of ``app.services.quantification`` and the Step 5 notes).
    ``label_ids`` scopes the metric to a subset of the dataset's labels; ``None`` means
    the metric applies to every label.
    """
    metric_key: str = Field(..., description="Registry key of the metric to compute (must "
                                             "exist in METRIC_REGISTRY).")
    params: dict = Field(default_factory=dict, description="Per-metric parameter dict. "
                                                           "Currently stored as forward-looking "
                                                           "metadata; not yet threaded into "
                                                           "metric computation.")
    label_ids: list[int] | None = Field(default=None, description="Label ids this metric is "
                                                                  "scoped to. None means all "
                                                                  "labels of the dataset.")

    @field_validator("metric_key")
    @classmethod
    def _metric_key_must_be_registered(cls, value: str) -> str:
        if value not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric key '{value}'. Known metrics: "
                             f"{sorted(METRIC_REGISTRY)}.")
        return value


class QuantificationProfile(BaseModel):
    """A named, per-dataset collection of metric selections.

    Mirrors the existing schema style (see ``schemas/database/labels.py``): a pydantic
    model with ``from_db`` to build from the ORM row and helpers to serialize back to the
    JSON column the row stores its entries in.
    """
    id: int | None = Field(default=None, description="The profile id (None before insert).")
    dataset_id: int = Field(..., description="The dataset this profile belongs to.")
    name: str = Field(..., description="Human-readable profile name.")
    is_default: bool = Field(default=False, description="Whether this is the dataset's default "
                                                        "profile (at most one per dataset).")
    entries: list[ProfileEntry] = Field(default_factory=list,
                                        description="Ordered list of metric selections.")

    @classmethod
    def from_db(cls, profile) -> "QuantificationProfile":
        """Build a schema from a ``QuantificationProfiles`` ORM row.

        The row stores its entries as a JSON list on ``profile.entries``; each element is
        validated back into a :class:`ProfileEntry` (so a stored unknown metric key would
        raise, surfacing registry drift).
        """
        raw_entries = profile.entries or []
        return cls(
            id=profile.id,
            dataset_id=profile.dataset_id,
            name=profile.name,
            is_default=bool(profile.is_default),
            entries=[ProfileEntry(**entry) for entry in raw_entries],
        )

    def entries_as_json(self) -> list[dict]:
        """Serialize ``entries`` to the plain list-of-dicts stored in the JSON column."""
        return [entry.model_dump() for entry in self.entries]

    def metric_keys(self) -> list[str]:
        """Unique metric keys referenced by this profile, preserving entry order."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.metric_key not in seen:
                seen.append(entry.metric_key)
        return seen
