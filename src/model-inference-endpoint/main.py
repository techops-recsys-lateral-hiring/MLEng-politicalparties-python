import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, filepath="../../data/Tweets.csv"):
        self.filepath = filepath
        self.load_data()
        self.vect_2()
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
        vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)
        return vectorizer.fit_transform(tweets).toarray()

    def vect_2(self):
        vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8)
        tweets = self.data.Tweet.apply(self.clean_text)
        self.vectorizer = vectorizer.fit(tweets.values)

    @staticmethod
    def label_encoder(parties):
        le = LabelEncoder()
        return le.fit_transform(parties)

    def preprocess(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values), self.label_encoder(self.data.Party.values)

mlflow.set_tracking_uri('file:///Users/cullywest/git/MLEng-politicalparties-python/data')
mlflow.set_experiment("MLflow")

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):

    model_version = 1
    model_name = "model_2024-03-15 11:48:19.976403"

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    vectorizer_name = "tfidf_vectorizer_2024_03_20T10:49:15"
    vectorizer_version = 1

    vectorizer = mlflow.sklearn.load_model(model_uri=f"models:/{vectorizer_name}/{vectorizer_version}")
    vectorized_data = vectorizer.transform(input_data.input_texts.split(','))

    preds = model.predict(vectorized_data)


    return str(preds)
    # load model
    # clean input text 
    # predict and return json

# outside code in terminal docker build + docker run + send a curl request