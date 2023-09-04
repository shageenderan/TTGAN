"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
import pickle5 as pickle

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data

def load_PSG(actuation='Random'):
  """Loads half of the PSG dataset. The data should have been already split in half
     and normalized accordingly (-mean/std)
  
  Args:
    - actuation: either 'Random' or 'Osc'
    
  Returns:
    - data: (total_samples, seq_len, n_features) -> (total_samples, 50, 99)
  """  
  if actuation not in ['Random', 'Osc']:
    raise NotImplementedError("No known actuation")
  data = np.load(f'data/PSG/{actuation}_Gripper/{actuation}_Gripper_Half.npy')
  return data

def load_PSC(scaler=None):
  """Loads half of the PSC dataset The data should have been already split in half.
     This function then normalizes accordingly (-mean/std)
  
  Args:
    - scaler: either None or the filepath of a previously saved scaler
    
  Returns:
    - data: (total_samples, seq_len, n_features) -> (total_samples, 50, 33)
  """
  train_data = load_pickle('data/PSC/rand_free_train_half.pkl')

  if scaler is None:
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    # with open("data/3DoF/rand_free_half_scaler.pkl", "wb") as f:
    #   pickle.dump(scaler, f)
  else:
    train_data = scaler.transform(train_data)

  # Timeseries
  train_data = to_timeseries(train_data, 50)

  # Shuffle
  np.random.shuffle(train_data)

  return train_data

def load_PSF(s='rand_free_30', axis='Train', no_samples=-1, scaler=None):
  """Loads raw PSF data
  
  Args:
    - s: one of: ["osc_free_30", "osc_tip_30", "osc_rand_30", "rand_tip_30", "rand_free_30", "rand_rand_60"]
    - axis: one of: ["Train", "Test"] to select whether to load training or testing data
    - no_samples: Number of samples to load. Use -1 (default) to load entire dataset
    - scaler: either None or the filepath of a previously saved scaler
    
  Returns:
    - data: (total_samples, seq_len, n_features) -> (total_samples, 50, 24)
  """
  # Loads raw PSF data based on the scenario 's'. Will returned a normalized
  ## timeseries (samples, seq_len, n_features) dataset.
  if s not in ['osc_free_30', "osc_tip_30", "osc_rand_30", "rand_tip_30", "rand_free_30", "rand_rand_60"]:
    raise NotImplementedError(f'No known actuation and/or scenario\nPlease use one of: {["osc_free_30", "osc_tip_30", "osc_rand_30", "rand_tip_30", "rand_free_30", "rand_rand_60"]}')

  print(f"Loading {s} {axis}ing data. To switch to {'testing' if axis=='Train' else 'training'} data, enter axis={'Test' if axis=='Train' else 'Train'}" )
  raw_data = load_pickle("data/PSF/new_soro_all.pkl")
  train = raw_data.xs(axis, axis=0, level=1, drop_level=True)

  train_data = _preprocess(train[s], no_samples=no_samples)

  # Normalize 
  if scaler is None:
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    # with open("data/osc_rand_scaler.pkl", "wb") as f:
    #   pickle.dump(scaler, f)
  else:
    train_data = scaler.transform(train_data)

  # Timeseries
  train_data = to_timeseries(train_data, 50)

  # Shuffle
  np.random.shuffle(train_data)

  return train_data #, scaler #, feature_scaler, label_scaler

def to_timeseries(X, time_steps = 50):
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=X,
          targets=None,
          sequence_length=time_steps,
          sequence_stride=1,
          shuffle=True,
          batch_size=1)
  temporal_X = []
  for seq in ds:
      data = seq
      temporal_X.append(data[0])
  temporal_X = np.array(temporal_X)
  return temporal_X

def _preprocess(data, scaler=None, no_samples=-1):
  # Features
  flex = data['Flex'][0]
  pressure = data['Pressure'][0]

  # Labels
  f1, f2 = data['Force']
  x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = data['PosX']
  y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = data['PosY']
      
  x = list(zip(flex, pressure, f1, f2, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10))
    
  # remove data to match samples
  if no_samples > 0:
    x = sample(np.array(x), no_samples)

  return np.array(x)

def load_pickle(filepath):
  with open(filepath,'rb') as f:
      x = pickle.load(f)
      return x

def sample(X, n_samples):
  np.random.seed(26)
  indexes = np.random.choice(len(X), len(X) - n_samples, replace=False)
  # print(indexes)
  return np.delete(X, indexes, 0) 

if __name__ == "__main__":
  pass