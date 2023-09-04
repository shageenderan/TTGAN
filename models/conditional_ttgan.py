# TransformerTimeGAN
"""
Base Conditional-TTGAN class implemented accordingly with:
Original base TimeGAN code can be found here: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor,float32
from tensorflow import data as tfdata
from tensorflow import config as tfconfig
from tensorflow import nn
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import GRU, LSTM, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.compat.v1.keras.layers import CuDNNGRU

import numpy as np
from tqdm import tqdm, trange

import tensorflow as tf
from keras import backend as K

from gan import Model as GAN

def make_net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type=='GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                      return_sequences=True,
                      name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model

class Conditional_TTGAN(GAN):
    def __init__(self, model_parameters, hidden_dim, seq_len, n_features, gamma, cond_vector_size):
        self.seq_len=seq_len
        self.n_features=n_features
        self.hidden_dim=hidden_dim
        self.cond_vector_size = cond_vector_size
        self.gamma=gamma
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator_aux=Generator(self.hidden_dim).build(input_shape=(self.seq_len, self.n_features+self.cond_vector_size))
        self.supervisor=Supervisor(self.hidden_dim).build(input_shape=(self.seq_len, self.hidden_dim))
        self.discriminator=Discriminator(self.hidden_dim).build(input_shape=(self.seq_len, self.hidden_dim+self.cond_vector_size))
        # self.recovery = Recovery(self.hidden_dim, self.n_features).build(input_shape=(self.hidden_dim, self.hidden_dim))
        # self.embedder = Embedder(self.hidden_dim).build(input_shape=(self.seq_len, self.n_features))
        self.embedder = Transformer(num_layers=3, hidden_dim=256, num_heads=8,
                         dff=512, output_units=self.hidden_dim, seq_len=self.seq_len, name='Embedder')
        self.recovery = Transformer(num_layers=3, hidden_dim=256, num_heads=8,
                         dff=512, output_units=self.n_features, seq_len=self.seq_len, name='Recovery')

        X = Input(shape=[self.seq_len, self.n_features], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_features+self.cond_vector_size], batch_size=self.batch_size, name='Conditioned_Noise')
        C = Input(shape=[self.seq_len, self.cond_vector_size], batch_size=self.batch_size, name='Conditional_Vector')

        # Concat = Concatenate(-1)

        #--------------------------------
        # Building the AutoEncoder
        #--------------------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X, outputs=X_tilde)

        #---------------------------------
        # Adversarial Supervise Architecture
        #---------------------------------
        E_hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_hat)

        # Conditional Info
        H_hat_Conditioned = concatenate([H_hat, C], name="Conditioned_H_hat")
        E_hat_Conditioned = concatenate([E_hat, C], name="Conditioned_E_hat")
        H_Conditioned = concatenate([H, C], name="Conditioned_H")        

        Y_fake = self.discriminator(H_hat_Conditioned)

        self.adversarial_supervised = Model(inputs=[Z,C],
                                       outputs=Y_fake,
                                       name='AdversarialSupervised')

        #---------------------------------
        # Adversarial architecture in latent space
        #---------------------------------
        Y_fake_e = self.discriminator(E_hat_Conditioned)

        self.adversarial_embedded = Model(inputs=[Z,C],
                                    outputs=Y_fake_e,
                                    name='AdversarialEmbedded')
        # ---------------------------------
        # Synthetic data generation
        # ---------------------------------
        X_hat = self.recovery(H_hat)
        self.generator = Model(inputs=Z,
                            outputs=X_hat,
                            name='FinalGenerator')

        # --------------------------------
        # Final discriminator model
        # --------------------------------
        Y_real = self.discriminator(H_Conditioned)
        self.discriminator_model = Model(inputs=[X,C],
                                         outputs=Y_real,
                                         name="RealDiscriminator")

        # ----------------------------
        # Define the loss functions
        # ----------------------------
        self._mse=MeanSquaredError()
        self._bce=BinaryCrossentropy()

    def get_summary(self):
        print("Blocks")
        print(self.embedder.summary())
        print(self.recovery.summary())
        print(self.generator_aux.summary())
        print(self.supervisor.summary())
        print(self.discriminator.summary())

        print("Main Models")
        print(self.autoencoder.summary())
        print(self.generator.summary())
        print(self.discriminator.summary())

    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            g_loss_s = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return g_loss_s

    @function
    def train_embedder(self,x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z, c):
        y_real = self.discriminator_model([x, c])
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        y_fake = self.adversarial_supervised([z, c])
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        y_fake_e = self.adversarial_embedded([z, c])
        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @function
    def train_generator(self, x, z, c, opt):
        with GradientTape() as tape:
            y_fake = self.adversarial_supervised([z,c])
            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            y_fake_e = self.adversarial_embedded([z, c])
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_hat = self.generator(z)
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @function
    def train_discriminator(self, x, z, c, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, z, c)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    def split_data(self, data):
        inputs = data[:, :self.n_features]
        labels = data[:, self.n_features:]

        assert labels.shape[-1] == self.cond_vector_size

        inputs.set_shape([None, self.n_features])
        labels.set_shape([None, self.cond_vector_size])
        return inputs, labels


    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data)
                                .shuffle(buffer_size=n_windows)
                                .repeat()
                                .map(self.split_data)
                                .batch(self.batch_size))

    def _generate_noise(self):
        return np.random.uniform(low=0, high=1, size=(self.batch_size, self.seq_len, self.n_features))

    def get_conditioned_batch_noise(self, condition):
        noise = self._generate_noise()
        conditioned_noise = np.concatenate((noise, condition), axis=-1)
        return conditioned_noise

    def train(self, data, train_steps):
        ## Embedding network training
        dataset = self.get_batch_data(data, n_windows=len(data))
        autoencoder_opt = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        embedder_time = ""
        embedder_loop = tqdm(range(train_steps), desc='Emddeding network training')
        for i in embedder_loop:
            X_, _ = next(dataset)
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)
            embedder_time = embedder_loop.format_interval(embedder_loop.format_dict['elapsed'])
            if i % 100 == 0:
                embedder_loop.set_postfix({"E_loss": step_e_loss_t0.numpy()})

        ## Supervised Network training
        dataset = self.get_batch_data(data, n_windows=len(data))
        supervisor_opt = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        supervisor_time = ""
        supervisor_loop = tqdm(range(train_steps), desc='Supervised network training')
        for i in supervisor_loop:
            X_, _ = next(dataset)
            step_g_loss_s = self.train_supervisor(X_, supervisor_opt)
            supervisor_time = supervisor_loop.format_interval(supervisor_loop.format_dict['elapsed'])
            if i % 100 == 0:
                supervisor_loop.set_postfix({"G_loss_S": step_g_loss_s.numpy()})

        ## Joint training
        generator_opt = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        embedder_opt = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        discriminator_opt = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        dataset = self.get_batch_data(data, n_windows=len(data))
        # noise = self.get_conditioned_batch_noise()

        _dataset = self.get_batch_data(data, n_windows=len(data))
        # _noise = self.get_conditioned_batch_noise()
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

        joint_time = ""
        train_loop = tqdm(range(train_steps), desc='Joint networks training')
        for i in train_loop:
            #Train the generator (k times as often as the discriminator)
            # Here k=2
            for _ in range(2):
                X_, y_ = next(dataset)
                Z_ = self.get_conditioned_batch_noise(y_)
                
                # --------------------------
                # Train the generator
                # --------------------------
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, y_, generator_opt)

                # --------------------------
                # Train the embedder
                # --------------------------
                step_e_loss_t0 = self.train_embedder(X_, embedder_opt)

            X, y = next(_dataset)
            Z = self.get_conditioned_batch_noise(y)
            step_d_loss = self.discriminator_loss(X, Z, y)
            if step_d_loss > 0.15:
                step_d_loss = self.train_discriminator(X, Z, y, discriminator_opt)
            joint_time = train_loop.format_interval(train_loop.format_dict['elapsed'])
            if i % 10 == 0:
                train_loop.set_postfix({"E_loss": step_e_loss_t0.numpy(), "D_loss": step_d_loss.numpy(), "G_loss_S": step_g_loss_s.numpy(), "G_loss_U": step_g_loss_u.numpy(), "G_loss_V": step_g_loss_v.numpy()})

        # return timings
        return embedder_time, supervisor_time, joint_time

    def save_model(self, path):
        # save autoencoder
        self.embedder.save(os.path.join(path, "encoder"))
        self.recovery.save(os.path.join(path, "decoder"))

        self.autoencoder.save(os.path.join(path, "autoencoder"))
        self.generator.save(os.path.join(path, "generator"))
        self.discriminator.save(os.path.join(path, "discriminator"))

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        noise = self.get_conditioned_batch_noise()
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = next(noise)
            records = self.generator(Z_)
            data.append(records)

        data = np.array(np.vstack(data))
        if len(data) > n_samples:
            data = data[:n_samples]
        return data


class Generator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self, input_shape):
        model = Sequential(name='Generator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim,
                         net_type=self.net_type)
        return model

class Discriminator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type=net_type

    def build(self, input_shape):
        model = Sequential(name='Discriminator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=1,
                         net_type=self.net_type)
        return model

class Supervisor(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim

    def build(self, input_shape):
        model = Sequential(name='Supervisor')
        model = make_net(model,
                         n_layers=2,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim)
        return model

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, hidden_dim, num_heads, dff, output_units,
               seq_len=50, rate=0.1, name="Model"):
    super(Transformer, self).__init__(name=name)

    # self.d_model = d_model
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.pos_encoding = positional_encoding(seq_len, self.hidden_dim)
    self.encoding = Dense(self.hidden_dim)

    self.enc_layers = [AttentionBlock(self.hidden_dim, num_heads, dff, rate, name=f"Attention_{i}")
                       for i in range(num_layers)]

    self.linear = Dense(output_units, activation='sigmoid')

  @tf.function
  def call(self, x, training=False, mask=None):
    #  encode x into hidden dim
    x = self.encoding(x)

    # add positional encoding
    x += self.pos_encoding

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    x = self.linear(x)

    return x  # (batch_size, input_seq_len, d_model)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
  
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


# from (https://www.tensorflow.org/text/tutorials/transformer)
@tf.function
def scaled_dot_product_attention(q, k, v, mask=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  @tf.function
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  @tf.function
  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Conv1D(dff, kernel_size=1, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Conv1D(d_model, kernel_size=1)  # (batch_size, seq_len, d_model)
  ])

class AttentionBlock(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1, name="Attention"):
    super(AttentionBlock, self).__init__(name=name)

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # self.dropout1 = tf.keras.layers.Dropout(rate)
    # self.dropout2 = tf.keras.layers.Dropout(rate)

  @tf.function
  def call(self, x, training=False, mask=None):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    # attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    # ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2