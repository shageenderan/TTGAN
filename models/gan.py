import tqdm
from joblib import dump, load
import pandas as pd
import tensorflow as tf
from tensorflow import config as tfconfig
from tensorflow.keras import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(**saving_utils.compile_args_from_training_config(training_config))
    restored_model.set_weights(weights)
    return restored_model

def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__=__reduce__

class Model():
    def __init__(self, model_parameters):
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            print("GPU available:", gpu_devices)

        self._model_parameters = model_parameters
        [self.batch_size, self.lr, self.noise_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    def define_gan(self):
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        return network.trainable_variables

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def model_name(self):
        return self.__class__.__name__

    def train(self, data, train_arguments):
        raise NotImplementedError

    def save(self, path):
        make_keras_picklable()
        try:
            dump(self, path)
        except:
            raise Exception('Please provide a valid path to save the model.')

    @classmethod
    def load(cls, path):
        synth = load(path)
        return synth

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(pd.DataFrame(records))
        return pd.concat(data)