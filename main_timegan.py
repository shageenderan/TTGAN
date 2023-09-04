"""TTGAN Codebase.

Modified from:
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from models.ttgan import TimeGAN as ttgan
from models.timegan import TimeGAN as timegan_ori

# 2. Data loading
from data_loading import sine_data_generation, load_PSG, load_PSC, load_PSF
# from load_condition_data import load_gripper_data as load_conditional_data

# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_gpu import predictive_score_metrics
from metrics.visualization_metrics import visualization


def revert_timeseries(X, time_steps = 50):
  head = X[0]
  tail = np.array([f[time_steps-1] for f in X])
  return np.concatenate((head, tail))

def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  elif args.data_name == 'PSF':
    ori_data = load_PSF(f'{args.actuation}')
  elif args.data_name == 'PSG':
    ori_data = load_PSG(args.actuation)
  elif args.data_name == 'PSC':
    ori_data = load_PSC()

  if args.timegan == 'ori':
    TimeGAN = timegan_ori
  elif args.timegan == 'ttgan':
    TimeGAN = ttgan
  print("Loaded TimeGAN:", args.timegan)
  
  # print(f'{args.data_name}:{args.actuation},{args.scenario}, {args.ratio} {np.shape(ori_data)} dataset is ready.')
  print(f'{args.data_name} {args.actuation if len(args.actuation) else ""} {np.shape(ori_data)} dataset is ready.')

    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['epochs'] = args.epochs
  parameters['batch_size'] = args.batch_size
  parameters['learning_rate'] = args.lr

  print("Parameters are: ", str(parameters))   
  # generated_data, timings = timegan(ori_data, parameters)  

  seq_len = np.shape(ori_data)[1]
  n_seq = np.shape(ori_data)[2]
  hidden_dim = args.hidden_dim
  gamma = 1
  noise_dim = np.shape(ori_data)[2]
  batch_size = args.batch_size
  learning_rate = args.lr
  # learning_rate = 5e-4

  gan_args = [batch_size, learning_rate, noise_dim]

  # if args.timegan == 'ori':
  #   timings = []
  #   generated_data = timegan_ori(ori_data, parameters)
  # else:
  #   tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
  #   timegan = TimeGAN(gan_args, hidden_dim, seq_len, n_seq, gamma)
  #   timings = timegan.train(ori_data, train_steps=args.iteration)

  #   # Save model
  #   timegan.save_model(os.path.join(args.output, "model"))

  #   generated_data = timegan.sample(len(ori_data))
  timegan = TimeGAN(gan_args, hidden_dim, seq_len, n_seq, gamma)
  # timings = timegan.train(ori_data, epochs=args.epochs)
  timings = timegan.train(ori_data, train_steps=args.iteration)


  # Save model
  if args.timegan != 'ttgan_time2vec' or  args.timegan != 'timegan_9':
    timegan.save_model(os.path.join(args.output, "model"))

  generated_data = timegan.sample(len(ori_data))

  print('Finish Synthetic Data Generation:', np.shape(generated_data), timings)
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = discriminative_score if len(discriminative_score) else 0
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   

  metric_results['predictive'] = predictive_score if len(predictive_score) else 0  

  ## Print discriminative and predictive scores
  print(np.mean(metric_results['predictive']), np.mean(metric_results['discriminative'])) 
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca', args.output)
  visualization(ori_data, generated_data, 'tsne', args.output)
  
  return ori_data, generated_data, metric_results, timings


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine', 'conditional', 'PSF', 'PSC', 'PSG'],
      default='stock',
      type=str)
  # only needed for PSF and PSG
  parser.add_argument(
      '--actuation',
      help="If loading PSG data then actuation must be either ['Random', 'Osc'].\nIf loading PSF data, actuation must be one of: ['osc_free_30', 'osc_tip_30', 'osc_rand_30', 'rand_tip_30', 'rand_free_30', 'rand_rand_60']",
      default='',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--lr',
      help='learning rate',
      default=5e-4,
      type=float)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--n_samples',
      help='number of samples to take from training set',
      default=-1,
      type=int)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iteration (should be optimized)',
      default=5000,
      type=int)
  parser.add_argument(
      '--epochs',
      help='Training epochs (should be optimized)',
      default=50,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--output',
      help='Output path to save data',
      default='output/temp',
      type=str)
  parser.add_argument(
      '--timegan',
      choices=['ori', 'ttgan'],
      default='ori',
      type=str)
  
  args = parser.parse_args() 
  args.output = os.path.join("output", args.output)

  # Create output directory
  # try:
  #   os.makedirs(args.output)
  # except FileExistsError:
  #   print(f"Skipping {args.output}")
  #   quit()
    # ans = input("Directory '{}' already exists, overwrite?(y/n):".format(args.output))
    # ans = ans.lower()
    # if ans != 'y' or ans != 'yes':
    #   quit()


  # Calls main function  
  ori_data, generated_data, metrics, timings = main(args)

  with open(os.path.join(args.output, "synthetic_dp_score.txt"), "w+") as f:
    # f.write('Discriminative Score - Mean: ' + str(np.round(np.mean(metrics['discriminative']),4)) + ', Std: ' + str(np.round(np.std(metrics['discriminative']),4)))
    f.write('\nPredictive Score - Mean: ' + str(np.round(np.mean(metrics['predictive']),4)) + ', Std: ' + str(np.round(np.std(metrics['predictive']),4)))
    f.write(f'\nAll Predictive Scores: {metrics["predictive"]}')
    f.write('\nDiscriminative Score - Mean: ' + str(np.round(np.mean(metrics['discriminative']),4)) + ', Std: ' + str(np.round(np.std(metrics['discriminative']),4)))
    f.write(f'\nAll Discriminative Scores: {metrics["discriminative"]}')
    f.write(f'\nTimings: {timings}' )
  np.save(os.path.join(args.output, "synthetic_data"), generated_data)