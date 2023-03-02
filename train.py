import os
import random
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

from config import config as opt
from data import CommonLitData
from model import CommonLitRegressionModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

writer = SummaryWriter(log_dir = os.path.join(opt.checkpoint_dir, 'tf_log'))

def score_model(model, dataloader):
    """
    :param model:
    :param dataloader:
    :return:
        res: RMSE
    """
    print('Model scoring was started')
    model.eval()
    dataloader.dataset.mode = 'eval'
    result = []
    targets = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            inputs = batch[0].to(opt.device)
            values = batch[1].to(opt.device)
            predicted = model(inputs)
            result.extend(torch.squeeze(predicted).cpu().numpy().tolist())
            targets.extend(values.cpu().numpy().tolist())
    rmse = mean_squared_error(targets, result, squared=False)
    return rmse

def train_epoch(epoch, model, dataloader, optimizer):
    print(f'start {epoch} epoch')
    for step, batch in tqdm(enumerate(dataloader)):
        inputs = batch[0].to(opt.device)
        values = batch[1].to(opt.device)
        predicted = model(inputs)
        loss = F.mse_loss(torch.squeeze(predicted), values)
        writer.add_scalar('loss', loss.item(), global_step=epoch * len(dataloader) + step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_model():
    data = CommonLitData(opt.train_file, opt.batch_size, opt.val_part, opt.device)
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    if opt.load_checkpoint_path:
        model = CommonLitRegressionModel(opt.regressor_dropout)
        model.load_state_dict(torch.load(opt.load_checkpoint_path))
    else:
        model = CommonLitRegressionModel(opt.regressor_dropout)
    model.to(opt.device)

    optimizer = Adam(model.parameters(), lr = opt.lr)
    
    for epoch in range(1, opt.n_epoch + 1):
        model = train_epoch(epoch, model, train_loader, optimizer)
        torch.save(model.state_dict(), join(opt.checkpoint_dir, f'weights/latest/latest.pth'))
        if epoch % opt.scoring_everyN_epoch == 0:
            rmse = score_model(model, val_loader)
            writer.add_scalar('rmse', rmse, global_step=epoch)
            print(f'rmse: {rmse}')
            torch.save(model.state_dict(), join(opt.checkpoint_dir, f'epoch{epoch}_RMSE={round(rmse, 5)}.pth'))
    return model
        
if __name__ == '__main__':
    train_model()