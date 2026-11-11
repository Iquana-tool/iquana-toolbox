from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from iquana_toolbox.schemas.database.labels import Label


@pytest.mark.parametrize("value", [0, 7])
def test_from_db_preserves_integer_label_values(value):
    row = SimpleNamespace(
        id=1,
        dataset_id=2,
        name="example",
        value=value,
        parent_id=None,
    )

    assert Label.from_db(row).value == value


def test_from_db_rejects_null_value():
    row = SimpleNamespace(
        id=1,
        dataset_id=2,
        name="example",
        value=None,
        parent_id=None,
    )

    with pytest.raises(ValidationError):
        Label.from_db(row)
