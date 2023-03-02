import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

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
        self._regressor = torch.load(model_path)

    def predict(self, text):
        inputs = self._tokenizer(text, return_tensors="pt")
        encoding = self._bert(**inputs)
        return self._regressor(encoding[0][:, -1]).item()

if __name__ == '__main__':
    model = CommonLitRegressionModel()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer("Some text", return_tensors="pt")
    encoding = encoder(**inputs)
    print(encoding[0][:, 0, :])
    print(model(encoding[0][:, 0, :]).item())
