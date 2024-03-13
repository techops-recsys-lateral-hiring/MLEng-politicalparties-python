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
    df = pd.DataFrame(data=data, index=[0,1])
    return df

@pytest.fixture
def data_loader(mocker, mock_df):
    mocker.patch('pandas.read_csv', return_value=mock_df)
    return DataLoader()

def test_load_data(mocker, mock_df):
    data_loader = DataLoader()
    mocker.patch.object(data_loader, 'data',  mock_df)
    data_loader.load_data()
    assert not data_loader.data.empty, "Data should not be empty after loading."

def test_remove_punct_and_digits(mock_df):
    text = mock_df['Tweet'].iloc[1]
    cleaned_text = DataLoader.remove_punct_and_digits(text)
    assert cleaned_text == "blohskdjfps"

def test_remove_urls(mock_df):
    text = mock_df['Tweet'].iloc[0]
    cleaned_text = DataLoader.remove_urls(text)
    assert cleaned_text == 'blah!23564 '

def test_clean_text(data_loader, mock_df):
    text = mock_df['Tweet'].iloc[0]
    cleaned_text = data_loader.clean_text(text)
    assert cleaned_text == "blah"
    
def test_preprocess(data_loader):
    features, labels = data_loader.preprocess()
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert isinstance(labels, np.ndarray), "Labels should be a numpy array"

