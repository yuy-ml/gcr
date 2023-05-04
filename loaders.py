import os.path as path
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np

from tqdm import tqdm
from itertools import islice
from dataclasses import dataclass

from torch.utils.data import Dataset

from utils import label2vec

matplotlib.use('Agg')

def get_audio_by_id(base_folder: str, id: int, sr=22050):
  id = f'{id:06}'
  subfolder = id[:3]
  filename = f'{base_folder}/{subfolder}/{id}.mp3'
  return librosa.load(filename, sr=sr)[0]


def retrieve_image(audio, n_fft, sr, win_length, hop_length, fmin, fmax):
  
  audio = np.pad(audio, (0, max(sr * 31 - audio.shape[0], 0)))
  spec = librosa.power_to_db(
      librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                     hop_length=hop_length, 
                                     win_length=win_length))
  
  fig = plt.figure(frameon=False)
  ax = fig.add_subplot(111)
  ax.set_position([0, 0, 1, 1])
  plt.axis('off')
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
  fig.patch.set_alpha(0)
  
  librosa.display.specshow(spec, ax=ax, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, fmin=fmin, fmax=fmax,
                            sr=sr)
  fig.canvas.draw()
  rgba_buf = fig.canvas.buffer_rgba()
  (w,h) = fig.canvas.get_width_height()
  plt.clf()
  plt.close('all')
  rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]

  
  return rgba_arr / 255
  
  
@dataclass
class MelImageDataset(Dataset):
  data: pd.DataFrame
  sr: int = 22050
  n_fft: int = 2048
  hop_length: int = 512
  win_length: int = n_fft
  suffix: int = None
  fmin: int = 50
  fmax: int = 14000
  data_dir: str = None
  
  x = None
  y = None
  dtemp = ".cache"
  progress_path = None
  
  def __restore_state__(self):
    try:
      with open(self.progress_path, 'r') as f:
        i = int(f.read())
    except OSError:
      i = 0
    return i
  
  def __save_state__(self, i):
    with open(self.progress_path, 'w') as f:
      f.write(str(i))
      
  def unload(self):
    self.x = np.memmap(path.join(self.cache_path, 'x.dat'),
                       shape=(self.data.shape[0], 480, 640, 3),
                       dtype=np.float32,
                       mode='r+')
    self.y = np.memmap(path.join(self.cache_path, 'y.dat'),
                       shape=(self.data.shape[0], 8),
                       dtype=np.float32,
                       mode='r+')
    
  def __process__(self, init_state):
    i = init_state
    for index, row in tqdm(islice(self.data.iterrows(), init_state, None, 1),
                           total=len(self.data), initial=init_state):
      if i % 100 == 0:
        self.__save_state__(i)
      
      try:
        audio = get_audio_by_id(self.data_dir, index, self.sr)
      except Exception:
        self.x[i] = np.zeros((480, 640, 3))
        self.y[i] = np.zeros(8)
        i += 1
        continue
      
      image = retrieve_image(audio, n_fft=self.n_fft,
                                    win_length=self.win_length,
                                    sr=self.sr,
                                    hop_length=self.hop_length,
                                    fmin=self.fmin,
                                    fmax=self.fmax)
      self.x[i] = image
      self.y[i] = label2vec(8)(int(row['track_genres']))
      i += 1
      
  def __shuffle__(self):
    print("Shuffling after first initialization")
    p = np.random.permutation(self.data.shape[0])
    self.x[:], self.y[:] = self.x[p], self.y[p]
    self.x.flush()
    self.y.flush()
    self.unload()
    
  def __init_cache__(self):
    self.cache_path = f"{self.dtemp}/genre-research_{self.suffix}"
    self.progress_path = f'{self.cache_path}/.progress'
    if path.exists(self.cache_path):
      mode = 'r+'
    else:
      os.makedirs(self.cache_path)
      mode = 'w+'
  
    self.x = np.memmap(path.join(self.cache_path, 'x.dat'),
                       shape=(self.data.shape[0], 480, 640, 3),
                       dtype=np.float32,
                       mode=mode)
    self.y = np.memmap(path.join(self.cache_path, 'y.dat'),
                       shape=(self.data.shape[0], 8),
                       dtype=np.float32,
                       mode=mode)
  
  def __post_init__(self):
    
    self.__init_cache__()
    init_state = self.__restore_state__()
    self.__process__(init_state)
    self.__save_state__(str(len(self.data)))
    
    
    if init_state != self.data.shape[0]:
      self.__shuffle__()
    
  def __len__(self):
    return self.data.shape[0]
  
  def __getitem__(self, index):
    x_cur = self.x[index]
    y_cur = self.y[index]
    
    return np.swapaxes(x_cur, 0, 2), y_cur