from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from text_loader.loader import DataLoader

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):

    vectorizer = mlflow.sklearn.load_model("data/models/vectorizer")
    vectorized_data = vectorizer.transform([DataLoader().clean_text(text=input_data.input_texts)])

    model = mlflow.sklearn.load_model("data/models/model")
    preds = model.predict(vectorized_data)

    return str(preds)