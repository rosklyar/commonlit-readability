import os
import torch

class Config:
    def __init__(self):
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.checkpoint_dir = 'checkpoints/commonlit_nn'
        self.load_checkpoint_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32
        self.val_part = 0.2
        self.n_epoch = 60
        self.lr = 0.0002
        self.scoring_everyN_epoch = 3
        self.regressor_dropout = 0.2

config = Config()
os.makedirs(os.path.join(config.checkpoint_dir, 'weights/latest'), exist_ok=True)