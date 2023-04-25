import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn import Module
from torch.optim import Optimizer
import os
import matplotlib.pyplot as plt

from typing import Tuple, Dict

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


def label2vec(n_classes: int) -> int:
  def inner(label):  
    v = np.zeros(n_classes)
    v[label] = 1
    return v
  return inner

def plot_csv(filepath: str, **kwargs) -> None:
  df = pd.read_csv(filepath)
  sns.lineplot(data=df, **kwargs)
  plt.show()
  
def save_state(model: Module, opt: Optimizer, epoch: int,
               losses: Dict, loss: float, acc: float, best_loss: float,
               best_acc: float, path: str) -> None:
  
  os.makedirs(path, exist_ok=True)
  pd.DataFrame(losses).to_csv(f'{path}/train_progress.csv', index=False)
  
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'loss': loss,
    'acc' : acc,
    'best_loss' : best_loss,
    'best_acc' : best_acc
    }, f'{path}/checkpoint.pt')
  
def restore_state(nn: Module, opt: Optimizer, path: str) -> Tuple[Module,
                                                                  Optimizer,
                                                                  int,
                                                                  float,
                                                                  float,
                                                                  float,
                                                                  float,
                                                                  Dict[str, list]]:
  
  try:
    checkpoint = torch.load(f'{path}/checkpoint.pt')
    
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    nn.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    train_process = pd.read_csv(f'{path}/train_progress.csv').to_dict('list')
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    best_loss = checkpoint['best_loss']
    best_acc = checkpoint['best_acc']
    
    print(f"Successfully loaded checkpoint: {best_loss} loss, {best_acc} acc")
    
    return nn, opt, epoch, loss, acc, best_loss, best_acc, train_process
  
  except FileNotFoundError as e:
    print(f"Couldn't load checkpoint: {e}")
    return nn, opt, -1, 1e18, 0, 1e18, 0, {}
  
class TrainLogger:
  def __init__(self, run: str) -> None:
    self.writer = SummaryWriter(run)
  
    layout = {
      "Performance": {
          "loss": ["Multiline", ["loss/train", "loss/val"]],
          "acc": ["Multiline", ["acc/train", "acc/val"]],
      },
    }  
    
    self.writer.add_custom_scalars(layout)
    
    
  def __add_tensorboard_record__(self, epoch: int, **dict):
    self.writer.add_scalar('loss/train', dict['val_loss'], epoch)
    self.writer.add_scalar('loss/val', dict['val_loss'], epoch)
    self.writer.add_scalar('acc/train', dict['acc'], epoch)
    self.writer.add_scalar('acc/val', dict['val_acc'], epoch)
    self.writer.add_scalar('lr', dict['lr'], epoch)
    
    
  def add(self, train_progress: Dict, **dict):
    self.__add_tensorboard_record__(**dict)
    
    for k, w in dict.items(): 
      if k not in train_progress:
        train_progress.update({k : []})
      train_progress[k].append(w)