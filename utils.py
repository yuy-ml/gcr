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
               losses: Dict, best_loss: int, best_acc: float, path: str) -> None:
  
  os.makedirs(path, exist_ok=True)
  pd.DataFrame(losses).to_csv(f'{path}/train_progress.csv', index=False)
  
  state = model.state_dict()
  state['epoch'] = epoch
  state['best_loss'] = best_loss
  state['best_acc'] = best_acc
  
  torch.save(opt.state_dict(), f'{path}/optimizer.pt')
  torch.save(state, f'{path}/model.pt')
  
def restore_state(nn: Module, opt: Optimizer, path: str) -> Tuple[Module,
                                                                  Optimizer,
                                                                  int,
                                                                  int,
                                                                  Dict[str, list]]:
  
  try:
    model_checkpoint = torch.load(f"{path}/model.pt")
    opt_checkpoint = torch.load(f"{path}/optimizer.pt")
    
    opt.load_state_dict(opt_checkpoint)
    nn.load_state_dict(model_checkpoint, strict=False)
    
    train_process = pd.read_csv(f'{path}/train_progress.csv').to_dict('list')
    epoch = model_checkpoint['epoch']
    best_loss = model_checkpoint['best_loss']
    best_acc = model_checkpoint['best_acc']
    print(f"Successfully loaded weights: {best_loss} loss, {best_acc} acc")
    
    return nn, opt, epoch, best_loss, best_acc, train_process
  
  except FileNotFoundError as e:
    print(f"Couldn't load checkpoint: {e}")
    return nn, opt, -1, 1e18, 0, {}
  
class LogWriter:
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

class DecayingCosineAnnealingLR(CosineAnnealingLR):
  def get_lr(self) -> float:
    lr = super().get_lr()
    decay_factor = np.exp([-0.05 * (self.last_epoch // (2 * self.T_max))])[0]
    lr[0] *= decay_factor
    return lr
  
  