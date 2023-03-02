import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizer, DistilBertModel


class CsvDataset(Dataset):
 
  def __init__(self, file_name, device):
    self._device = device
    self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    self._bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self._bert.to(device)
    price_df = pd.read_csv(file_name)
 
    x = price_df.iloc[:, 3].values
    y = price_df.iloc[:, 4].values
 
    self.x_train = x
    self.y_train = torch.tensor(y, dtype=torch.float32).to(device)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self, idx):
    inputs = self._tokenizer(self.x_train[idx], return_tensors="pt").to(self._device)
    encoding = self._bert(**inputs)[0][:, -1].to(self._device)
    return encoding, self.y_train[idx]
  
class CommonLitData():

    def __init__(self, dataset_file, batch_size, val_split, device):
        self.batch_size = batch_size
        # load *.csv
        self.dataset = CsvDataset(dataset_file, device)
        # split train.csv into train and validation
        val_length = int(len(self.dataset) * val_split)
        train_data, validation_data = random_split(self.dataset, [len(self.dataset) - val_length, val_length])
        # create train and validation dataloaders for pandas dataframe
        self._train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self._val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    def get_train_loader(self):
        return self._train_loader

    def get_val_loader(self):
        return self._val_loader