# Codebase for "TTGAN"

Authors: Shageenderan Sapai, Loo Junn Yong, Ding Ze Yang, Tan Chee Pin, Vishnu Monn Baskaran and Surya Girinatha Nurzaman
 
Paper Link: A Deep Learning Framework for Soft Robots with Synthetic Data https://doi.org/10.1089/soro.2022.0188

Reference: 
@article{sapai2023deep,
  title={A Deep Learning Framework for Soft Robots with Synthetic Data},
  author={Sapai, Shageenderan and Loo, Junn Yong and Ding, Ze Yang and Tan, Chee Pin and Baskaran, Vishnu Monn and Nurzaman, Surya Girinatha},
  journal={Soft Robotics},
  year={2023},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}


Contact: shageenderan.sapai@monash.edu

This directory contains implementations of TTGAN framework for synthetic time-series data generation
using one synthetic dataset and three self-curated soft robotic datasets.

-   Sine data: Synthetic
-   PSF Data: Time-series data collected from a pneumatic soft finger (PSF) platform. Consists of 2 featues (pressure + flex) and 22 labels (x,z resultant force and 10 2D markers) 
-   PSG: Time-series data collected from a pneumatic soft gripper (PSG) platform. This is an extension of the PSF platform and constitues 3 individual PSF's held together at equidistant points. Consists of 6 features (3 * (pressure + flex)) and 93 labels (x,y,z resultant force and 3*10 3D markers) 
-   PSC: Time-series data collected from a pneumatic soft conituum (PSC) body platform. This platform is capable of non-planar and 3D bending. Consists of 6 features (3 * (pressure + flex)) and 90 labels (3*10 3D markers).

To run the pipeline for training and evaluation on TTGAN framwork, simply run 
python3 -m main_timegan.py or see jupyter-notebook tutorial of TTGAN in tutorial_timegan.ipynb.

### Code explanation

(1) data_loading.py
- Transform raw soft robotic time-series data to preprocessed time-series data
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

(4) main_timegan.py
- Generate synthetic data and report discriminative and predictive scores for the dataset and t-SNE and PCA analysis

(5) utils.py
- Some utility functions for metrics.

### Command inputs:

-   data_name: sine, PSF, PSC or PSG
-   actuation: (PSG) 'Random' || 'Osc' or (PSF) 'osc_free_30' || 'osc_tip_30' || 'osc_rand_30' || 'rand_tip_30' || 'rand_free_30' || 'rand_rand_60'
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
$ python main_timegan.py --timegan ttgan 
--data_name PSF --actuation rand_free_30 --output ttgan/PSF/random
--batch_size 256 --seq_len 100 --module gru
--iteration 10000 --metric_iteration 5 
```

### Outputs

-   ori_data: original data
-   generated_data: generated synthetic data
-   metric_results: discriminative and predictive scores
-   visualization: PCA and tSNE analysis
