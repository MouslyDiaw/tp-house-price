"""Test dataset."""

from src.make_dataset import load_data

# load data
data = load_data(dataset_name="house_prices")


def test_shape():
    """Test data shape."""
    nrows, ncols = data.shape

    assert nrows >= 1460
    assert ncols == 81


def test_saleprice():
    """Target feature."""
    assert sum(data["SalePrice"].isnull()) == 0
    assert sum(data["SalePrice"] <= 0) == 0

