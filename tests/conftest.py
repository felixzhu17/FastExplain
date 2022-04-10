import pytest
import pandas as pd

@pytest.fixture(scope="session")
def test_csv():
    return pd.read_csv("titanic.csv")
