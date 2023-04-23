import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

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
  
def save_state(model, opt, epoch, losses, best_loss):
  
  os.makedirs('checkpoint', exist_ok=True)
  pd.DataFrame(losses).to_csv('checkpoint/train_progress.csv', index_label='id', index=False)
  
  state = model.state_dict()
  state['epoch'] = epoch
  state['best_loss'] = best_loss
  
  torch.save(opt.state_dict(), 'checkpoint/optimizer.pt')
  torch.save(state, 'checkpoint/model.pt')
  
def restore_state(model, opt, path):
  try:
    dict = torch.load(f"{path}/model.pt")
    opt.load_state_dict(torch.load(f"{path}/optimizer.pt"))
    model.load_state_dict(dict, strict=False)
    train_process = pd.read_csv('checkpoint/train_progress.csv').to_dict('list')
    return model, opt, dict['epoch'], dict['best_loss'], train_process
  except FileNotFoundError as e:
    print(f"Couldn't load checkpoint: {e}")
    return model, opt, -1, 1e18, {}
  
def init_tensorboard_writer(writer):
  layout = {
    "Performance": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
        "acc": ["Multiline", ["acc/train", "acc/val"]],
    },
  }
  
  writer.add_custom_scalars(layout)
  
  return writer
  
def add_tensorboard_record(writer, epoch, **dict):

  writer.add_scalar('loss/train', dict['val_loss'], epoch)
  writer.add_scalar('loss/val', dict['val_loss'], epoch)
  writer.add_scalar('acc/train', dict['acc'], epoch)
  writer.add_scalar('acc/val', dict['val_acc'], epoch)
  writer.add_scalar('lr', dict['lr'], epoch)
  
def add_record(train_progress, writer, **dict):
  for k, w in dict.items(): 
    if k not in train_progress:
      train_progress.update({k : []})
    train_progress[k].append(w)

  add_tensorboard_record(writer, **dict)

class DecayingCosineAnnealingLR(CosineAnnealingLR):
  def get_lr(self) -> float:
    lr = super().get_lr()
    decay_factor = np.exp([-0.05 * (self.last_epoch // (2 * self.T_max))])[0]
    lr[0] *= decay_factor
    return lr
  
  