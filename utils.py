import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR

def label2vec(n_classes):
  def inner(label):  
    v = np.zeros(n_classes)
    v[label] = 1
    return v
  return inner

def plot_csv(filepath, **kwargs):
  df = pd.read_csv(filepath)
  sns.lineplot(data=df, **kwargs)
  plt.show()
  
def save_state(model, epoch, losses, best_loss):
  
  os.makedirs('checkpoint', exist_ok=True)
  pd.DataFrame(losses).to_csv('checkpoint/train_process.csv')
  
  state = model.state_dict()
  state['epoch'] = epoch
  state['best_loss'] = best_loss
  
  torch.save(state, 'checkpoint/checkpoint.pt')
  
def restore_state(model, path):
  try:
    dict = torch.load(path)
    model.load_state_dict(dict)
    train_process = pd.read_csv('checkpoint/train_process.csv').to_dict()
    return model, dict['epoch'], dict['best_loss'], train_process
  except FileNotFoundError:
    print("Checkpoint does not exist")
    return model, 0, 1e18, {'epoch' : [], 'loss_type' : [], 'loss' : [], 'lr' : []}
  
def add_record(train_progress, loss_type, loss, epoch, lr):
  train_progress['epoch'].append(epoch)
  train_progress['loss_type'].append(loss_type)
  train_progress['loss'].append(loss.item())
  train_progress['lr'].append(lr)

class DecayingCosineAnnealingLR(CosineAnnealingLR):
  def get_lr(self) -> float:
    lr = super().get_lr()
    decay_factor = np.exp([-0.2 * self.last_epoch / self.T_max])[0]
    lr[0] *= decay_factor
    return lr
  
  