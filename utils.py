import numpy as np

def label2vec(n_classes):
  def inner(label):  
    v = np.zeros(n_classes)
    v[label] = 1
    return v
  return inner