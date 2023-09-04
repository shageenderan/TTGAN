import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam


def predictive_score_metrics (ori_data, synthetic_data):
    real_data = np.array(ori_data)[:len(synthetic_data)]

    seq_len = real_data.shape[1]
    dims = real_data.shape[-1]

    real_test_data = real_data[:, :seq_len-1, :]
    real_test_label = real_data[:, -1, :]

    
    synthetic_train = synthetic_data[:, :seq_len-1, :]
    synthetic_label = synthetic_data[:, -1, :]

    model = Sequential([GRU(12, input_shape=(seq_len-1, dims)),
                        Dense(dims)])

    model.compile(optimizer=Adam(),
                  loss=MeanAbsoluteError(name='MAE'),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    synthetic_result = model.fit(x=synthetic_train,
                                y=synthetic_label,
                                epochs=100,
                                batch_size=128,
                                verbose=0)

    performance = model.evaluate(real_test_data, real_test_label, verbose=0)
    # print(performance)
    return performance[1]