from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    # load model
    # clean input text 
    # predict and return json

# outside code in terminal docker build + docker run + send a curl request