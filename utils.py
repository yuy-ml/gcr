import numpy as np
import pandas as pd
import seaborn as sns
import json
import torch
import os
import matplotlib.pyplot as plt

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
  pd.DataFrame(losses).to_csv('checkpoint/losses.csv')
  
  state = model.state_dict()
  state['epoch'] = epoch
  state['best_loss'] = best_loss
  
  torch.save(state, 'checkpoint/checkpoint.pt')
  
def restore_state(model, path):
  try:
    dict = torch.load(path)
    model.load_state_dict(dict)
    return model, dict['epoch'], dict['best_loss'], pd.read_csv('checkpoint/losses.csv').to_dict()
  except FileNotFoundError:
    print("Checkpoint does not exist")
    return model, 0, 1e18, {}
  