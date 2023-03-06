# Codebase for "TTGAN"

Authors: Shageenderan Sapai

Reference: 
 
Paper Link: 

Contact: shageenderan.sapai@monash.edu

This directory contains implementations of TTGAN framework for synthetic time-series data generation
using one synthetic dataset and two real-world datasets.

-   Sine data: Synthetic
-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
-   Energy data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

To run the pipeline for training and evaluation on TTGAN framwork, simply run 
python3 -m main_timegan.py or see jupyter-notebook tutorial of TTGAN in tutorial_timegan.ipynb.

### Code explanation

(1) data_loading.py
- Transform raw time-series data to preprocessed time-series data (Googld data)
- Generate Sine data

(2) Metrics directory  
  (a) visualization_metrics.py
  - PCA and t-SNE analysis between Original data and Synthetic data   
  (b) discriminative_metrics.py
  - Use Post-hoc RNN to classify Original data and Synthetic data  
  (c) predictive_metrics.py
  - Use Post-hoc RNN to predict one-step ahead (last feature)  

(3) Models directory  
  (a) timegan.py
  - Original timegan implementation as in https://github.com/jsyoon0823/TimeGAN  
  (b) ttgan.py
  - Transformer timegan implementation  
  (c) ttgan_time2vec.py
  - Transformer timegan implementation using time2vec embedding+pos_embeddings  
  (d) timegan_9.py
  - Transformer timegan implementation using only time2vec embedding  

(4) main_timegan.py
- Report discriminative and predictive scores for the dataset and t-SNE and PCA analysis

(5) utils.py
- Some utility functions for metrics and timeGAN.

### Command inputs:

-   data_name: sine, stock, or energy
-   seq_len: sequence length
-   module: gru, lstm, or lstmLN
-   hidden_dim: hidden dimensions
-   num_layers: number of layers
-   iterations: number of training iterations
-   batch_size: the number of samples in each batch
-   metric_iterations: number of iterations for metric computation

Note that network parameters should be optimized for different datasets.

### Example command

```shell
$ python3 main_timegan.py --data_name stock --seq_len 24 --module gru
--hidden_dim 24 --num_layer 3 --iteration 50000 --batch_size 128 
--metric_iteration 10
```

### Outputs

-   ori_data: original data
-   generated_data: generated synthetic data
-   metric_results: discriminative and predictive scores
-   visualization: PCA and tSNE analysis
