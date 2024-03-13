import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath="data/Tweets.csv"):
        self.filepath = filepath
        self.load_data()
    
    def load_data(self):
        """Loads data from a CSV file."""
        self.data = pd.read_csv(self.filepath)

    @staticmethod
    def remove_punct_and_digits(text):
        remove_chars = string.punctuation + string.digits  # Define characters to remove
        translator = str.maketrans('', '', remove_chars)  # Create a translation table
        return text.translate(translator)
    
    @staticmethod
    def remove_urls(text):
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def clean_text(self, text):
        text = str(text).lower()
        text = self.remove_urls(text)
        text = self.remove_punct_and_digits(text)
        return text.strip()

    @staticmethod
    def vectorize_text(tweets):
        vectorizer = TfidfVectorizer (max_features=2500, min_df=1, max_df=0.8)
        return vectorizer.fit_transform(tweets).toarray()

    @staticmethod
    def label_encoder(parties):
        le = LabelEncoder()
        return le.fit_transform(parties)

    def preprocess(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values), self.label_encoder(self.data.Party.values)
    
