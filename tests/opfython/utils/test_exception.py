import pytest

from opfython.utils import exception


@pytest.mark.parametrize(
    "error",
    [
        exception.Error("Error", "error"),
        exception.ArgumentError("error"),
        exception.BuildError("error"),
        exception.SizeError("error"),
        exception.TypeError("error"),
        exception.ValueError("error"),
    ],
)
def test_package_exceptions_are_raiseable(error):
    with pytest.raises(exception.Error):
        raise error

    assert str(error) == "error"
