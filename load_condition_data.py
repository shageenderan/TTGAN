import os
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from matplotlib import pyplot as plt
import pickle5 as pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


ORI_DATA_DIR = 'data'

def _to_timeseries(X, time_steps = 50):
  dataX = []
  for i in range(0, len(X)):
      _x = [list(X[i])] * time_steps
      dataX.append(_x)

  return np.array(dataX)
def sample(X, n_samples):
  np.random.seed(0)
  indexes = np.random.choice(len(X), len(X) - n_samples, replace=False)
#   print(indexes)
  return np.delete(X, indexes, 0)

def load_gripper_scenario(scenario):
  # scenario == 0 = osc_tip
  # scenario == 1 = osc_free
  # scenario == 2 = rand_tip
  # scenario == 3 = rand_free
  if scenario == 0:
    print("Loading Osc Tip")
    data = np.load(f'data/gripper/Normalized (mean+std)/half/osc_tip_half.npy')
  elif scenario == 1:
    print("Loading Osc Free")
    data = np.load('data/gripper/Normalized (mean+std)/half/osc_free_half.npy')
  elif scenario == 2:
    print("Loading Rand Tip")
    data = np.load('data/gripper/Normalized (mean+std)/half/rand_tip_half.npy')
  else:
    print("Loading Rand Free")
    data = np.load(f'data/gripper/Normalized (mean+std)/half/rand_free_half.npy')
    
  np.random.shuffle(data)
  return data

def load_gripper_data(corrupt_scenario):
  np.random.seed(0)
  free_bending = [(1, 'osc_free'), (3, 'rand_free')]
  tip_contact = [(0, 'osc_tip'), (2, 'rand_tip')]
  # corrupt_scenario = scenario to remove
  cat_enc = OneHotEncoder(handle_unknown = 'ignore')
  # train_labels.append([train_scenario]*len(train_data[i]))
  #  train_labels = cat_enc.fit_transform(train_labels.reshape(-1,1)).toarray()
  data = []
  labels = []
  if corrupt_scenario == 'free_bending':
    for i, (code, scenario) in enumerate(free_bending):
      to_add = load_gripper_scenario(code)
      to_add = sample(to_add, len(to_add)//2)

      data.append(to_add)
      labels.append([scenario]*len(data[i]))

    for i, (code, scenario) in enumerate(tip_contact):
      to_add = load_gripper_scenario(code)
      data.append(to_add)
      labels.append([scenario]*len(data[i+2]))
    

  elif corrupt_scenario == 'tip_contact':
    for i, (code, scenario) in enumerate(tip_contact):
      to_add = load_gripper_scenario(code)
      to_add = sample(to_add, len(to_add)//2)

      data.append(to_add)
      labels.append([scenario]*len(data[i]))

    for i, (code, scenario) in enumerate(free_bending):
      to_add = load_gripper_scenario(code)
      data.append(to_add)
      labels.append([scenario]*len(data[i+2]))
    
  data = np.vstack(data)
  _labels = np.concatenate((labels), axis=0)
  cat_enc.fit_transform(_labels.reshape(-1,1))

  labels = [_to_timeseries(cat_enc.transform(np.reshape(l, (-1,1))).toarray()) for l in labels]
  labels = np.concatenate((labels), axis=0)
  return data, labels, cat_enc