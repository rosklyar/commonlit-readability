import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from fastapi import FastAPI
from pydantic import BaseModel, constr

class CommonLitRegressionModel(nn.Module):
    
    def __init__(self, dropout=0.2):
        super(CommonLitRegressionModel, self).__init__()
        self._regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features=768, out_features=1))
    
    def forward(self, x):
        return self._regressor(x)

class CommonLitInference():
    
    def __init__(self, model_path):
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self._regressor = CommonLitRegressionModel()
        self._regressor.load_state_dict(torch.load(model_path,  map_location=torch.device('cpu')))
        self._regressor.eval()

    def predict(self, text):
        inputs = self._tokenizer(text, return_tensors="pt")
        encoding = self._bert(**inputs)
        return self._regressor(encoding[0][:, -1]).item()

class TextIn(BaseModel):
    text: constr(min_length=1)

regressor = CommonLitInference('commonlit-model.pth')
app = FastAPI()


@app.post("/score")
def score(request_in: TextIn):
    return regressor.predict(request_in.text)