import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torchvision.models import resnet18
from loaders import retrieve_image, get_audio_by_id

model = resnet18()

torch.manual_seed(69)
  
model.fc = torch.nn.Sequential(
  torch.nn.Dropout(p=0.2, inplace=True),
  
  torch.nn.Linear(in_features=512,
                  out_features=256),
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.ReLU(inplace=True),
  
  torch.nn.BatchNorm1d(256),

  torch.nn.Linear(in_features=256,
                  out_features=128),
  torch.nn.Dropout(p=0.2, inplace=True),
  torch.nn.ReLU(inplace=True),
  
  torch.nn.BatchNorm1d(128),
  
  torch.nn.Linear(in_features=128,
                  out_features=32),
  torch.nn.Dropout(p=0.2, inplace=True),
  
  torch.nn.Linear(in_features=32,
                  out_features=8))

DATA_DIR='data/fma_small'

selected = pd.read_csv('data/selected.csv', index_col='track_id')
device = torch.device('mps')

model.to(device)

model.load_state_dict(torch.load(f'best_acc.pt', map_location=device))
model.eval()
batch_size = 256

probs = [[] for _ in range(8)]

rows = list(selected.iterrows())

for i in range(0, selected.shape[0], batch_size):
  print(f"Started processing batch {i // batch_size} of {selected.shape[0] // batch_size}")
  
  batch = []
  for index, row in tqdm(rows[i: i + batch_size], total=batch_size):
    try:
      audio = get_audio_by_id(DATA_DIR, index)
      image = retrieve_image(audio, sr=44100, win_length=1380, hop_length=345,
                                    n_fft=2048, fmin=50, fmax=14000)
    except Exception as e:
      print(e)
      image = np.zeros((480, 640, 3))
    
    image = np.swapaxes(image, 0, 2)
    
    batch.append(image)
    
  batch = np.array(batch, dtype=np.float32)
  with torch.no_grad():
    inputs = torch.tensor(batch, device=device)
    inputs.to(device)
    outputs = model(inputs)
    del inputs
    result = torch.nn.Softmax()(outputs)
    for b in range(min(batch_size, result.size(dim=0))):
      for i in range(result.size(dim=1)):
        probs[i].append(np.copy(result[b][i].cpu().numpy()))
    
for i in range(8):
  selected[str(i)] = probs[i]
  
selected.to_csv("inferenced.csv")