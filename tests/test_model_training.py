import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import sys #TODO fix path
sys.path.append('src/text_loader')
from loader import DataLoader

@pytest.fixture
def mock_df() -> pd.DataFrame:
    data = {
        "Party": ['Republican', 'Democrat'],
        "Tweet": ['blah!23564 https://hello.com', 'bloh@skdjfps']
    }
    df = pd.DataFrame(data=data)
    return df

def test_load_data(mocker, mock_df):
    data_loader = DataLoader()
    mocker.patch.object(data_loader, 'data',  mock_df)
    data_loader.load_data()
    assert not data_loader.data.empty, "Data should not be empty after loading."
    
def test_preprocess(mocker, mock_df: pd.DataFrame):
    data_loader = DataLoader()
    mocker.patch.object(data_loader, 'data',  mock_df)
    data_loader.preprocess()
    expected_df = pd.DataFrame(
        data={
            "Party": ['republican', 'democrat'],
            "Tweet": ['blah', 'blohskdjfps']
        },
        index=[0, 1],
    )
    assert_frame_equal(data_loader.data, expected_df)

