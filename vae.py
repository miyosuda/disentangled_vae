# -*- coding: utf-8 -*-
#
# VAE implementation is based on Jan Hendrik Metzen's code
# https://jmetzen.github.io/2015-11-27/vae.html

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave
from PIL import Image, ImageDraw

CHECKPOINT_DIR = 'checkpoints'
n_samples = 32 * 32 * 6 * 40
learning_rate = 5e-4
batch_size = 64
epoch_size = 100

#np.random.seed(0)
#tf.set_random_seed(0)

class Shape(object):
  def __init__(self, x, y, scale, rotate):
    self.x = x
    self.y = y
    self.scale = scale
    self.rotate = rotate

    
def generate_image(image_w, image_h, x, y, scale, rotate):
  # 8bitで画像生成
  im = Image.new("L", (image_w, image_h))

  # 縦長の楕円を中心に描画
  ex = image_w * scale
  ey = image_h * scale * 2.0

  # bounding box
  bbox =  (image_w/2 - ex/2,
             image_h/2 - ey/2,
             image_w/2 + ex/2,
             image_h/2 + ey/2)
  draw = ImageDraw.Draw(im)
  draw.ellipse(bbox, fill=255)
    
  # 回転
  im = im.rotate(360 * rotate)

  # 移動
  dx = (-x + 0.5) * image_w
  dy = (-y + 0.5) * image_h
  im = im.transform(im.size, Image.AFFINE, (1,0,dx,0,1,dy), Image.BILINEAR)
  del draw
  
  #0~255値のndarrayに
  im_arr = np.asarray(im)

  #0.0~1.0のfloat値のndarrayに
  im_arr = im_arr.astype(np.float32)
  im_arr = np.multiply(im_arr, 1.0 / 255.0)

  # reshape
  im_arr = im_arr.reshape((4096))
  return im_arr


def prepare_shapes():
  """ 学習用に全パターンの画像パラメータを用意する """
  shapes = []
  for x in np.linspace(0.2, 0.8, 32):
    for y in np.linspace(0.2, 0.8, 32):
      for scale in np.linspace(0.1, 0.2, 6):
        for rotate in np.linspace(0.0, 1.0, 40):
          shapes.append(Shape(x, y, scale, rotate))
  return shapes


def get_batch_images(shapes, batch_size, pos):
  """ 学習用の画像バッチを生成 """
  batch = []
  for i in range(batch_size):
    shape = shapes[pos+i]
    x = shape.x
    y = shape.y
    scale = shape.scale
    rotate = shape.rotate
    img = generate_image(64, 64, x, y, scale, rotate)
    batch.append(img)
  return batch


def get_random_images(size):
  """ 結果確認用にランダム画像を生成 """
  images = []
  for i in range(size):
    x = np.random.uniform(0.2, 0.8)
    y = np.random.uniform(0.2, 0.8)
    scale = np.random.uniform(0.1, 0.2)
    rotate = np.random.uniform(0.0, 1.0)
    img = generate_image(64, 64, x, y, scale, rotate)
    images.append(img)
  return images


def xavier_init(fan_in, fan_out, constant=1): 
  """ Xavier initialization of network weights"""
  # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  low  = -constant * np.sqrt(6.0/(fan_in + fan_out)) 
  high =  constant * np.sqrt(6.0/(fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out), 
                           minval=low,
                           maxval=high, 
                           dtype=tf.float32)


class VariationalAutoencoder(object):
  """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
  
  This implementation uses probabilistic encoders and decoders using Gaussian 
  distributions and realized by multi-layer perceptrons. The VAE can be learned
  end-to-end.
  
  See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
  """
  def __init__(self,
               network_architecture,
               learning_rate,               
               batch_size):
    
    self.network_architecture = network_architecture
    self.learning_rate        = learning_rate
    self.batch_size           = batch_size
    
    # tf Graph input (batch x 4096)
    self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
    
    # Create autoencoder network
    self._create_network()
    
    # Define loss function based variational upper-bound and 
    # corresponding optimizer
    self._create_loss_optimizer()
    
    # Initializing the tensor flow variables
    init = tf.global_variables_initializer()
    
    # Launch the session
    self.sess = tf.InteractiveSession()
    self.sess.run(init)

  def _create_network(self):
    # Initialize autoencode network weights and biases
    network_weights = self._initialize_weights(**self.network_architecture)

    # Use recognition network to determine mean and 
    # (log) variance of Gaussian distribution in latent space
    # Encoderの出力が、"z_mean"と"z_log_sigma_sq"
    # それぞれn_z=20個ずつの出力となる　
    self.z_mean, self.z_log_sigma_sq = \
      self._recognition_network(network_weights["weights_recog"], 
                                network_weights["biases_recog"])

    # Draw one sample z from Gaussian distribution
    # n_zは潜在変数の次元数: 20
    n_z = self.network_architecture["n_z"]
    # mean=0.0, stddev=1.0

    # バッチサイズを得る
    batch_sz = tf.shape(self.x)[0]
    # randomのshapeを動的に作る
    eps_shape = tf.stack([batch_sz, n_z])
    
    #eps = tf.random_normal( (self.batch_size, n_z), 0, 1, dtype=tf.float32 )
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    
    # z = mu + sigma * epsilon
    self.z = tf.add(self.z_mean, 
                    tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    # Decoderの出力が"x_reconstr_mean"
    self.x_reconstr_mean = \
      self._generator_network(network_weights["weights_gener"],
                              network_weights["biases_gener"])
    
  def _initialize_weights(self,
                          n_hidden_recog_1, # 1200
                          n_hidden_recog_2, # 1200
                          n_hidden_gener_1, # 1200
                          n_hidden_gener_2, # 1200
                          n_hidden_gener_3, # 1200
                          n_input,          # 4096
                          n_z):             # 10
    all_weights = dict()
    
    all_weights['weights_recog'] = {
      'h1'           : tf.Variable(xavier_init(n_input,          n_hidden_recog_1)),
      'h2'           : tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
      'out_mean'     : tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
      'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
    
    all_weights['biases_recog'] = {
      'b1'           : tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
      'b2'           : tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
      'out_mean'     : tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
      'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
    
    all_weights['weights_gener'] = {
      'h1'           : tf.Variable(xavier_init(n_z,              n_hidden_gener_1)),
      'h2'           : tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
      'h3'           : tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
      'out_mean'     : tf.Variable(xavier_init(n_hidden_gener_3, n_input)),
      'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_3, n_input))}
    
    all_weights['biases_gener'] = {
      'b1'           : tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
      'b2'           : tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
      'b3'           : tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
      'out_mean'     : tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
      'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
    
    return all_weights
      
  def _recognition_network(self, weights, biases):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x,  weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    z_mean         = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                            biases['out_log_sigma'])
    return (z_mean, z_log_sigma_sq)

  def _generator_network(self, weights, biases):
    # Generate probabilistic decoder (decoder network), which
    # maps points in latent space onto a Bernoulli distribution in data space.
    # The transformation is parametrized and can be learned.
    layer_1 = tf.tanh(tf.add(tf.matmul(self.z,  weights['h1']), biases['b1']))
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['out_mean']), 
                                           biases['out_mean']))
    return x_reconstr_mean
      
  def _create_loss_optimizer(self):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #   of the input under the reconstructed Bernoulli distribution 
    #   induced by the decoder in the data space).
    #   This can be interpreted as the number of "nats" required
    #   for reconstructing the input when the activation in latent
    #   is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)
    reconstr_loss = -tf.reduce_sum(self.x     * tf.log(1e-10 +     self.x_reconstr_mean) +
                                   (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                                   1)
    
    # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
    ##  between the distribution in latent space induced by the encoder on 
    #   the data and some prior. This acts as a kind of regularizer.
    #   This can be interpreted as the number of "nats" required
    #   for transmitting the the latent space distribution given
    #   the prior.
    beta = 4.0
    
    latent_loss = beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                              - tf.square(self.z_mean) 
                                              - tf.exp(self.z_log_sigma_sq), 1)
    
    self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch

    self.optimizer = tf.train.AdagradOptimizer(
      learning_rate=self.learning_rate).minimize(self.cost)
    
  def partial_fit(self, X):
    """Train model based on mini-batch of input data.
    
    Return cost of mini-batch.
    """
    opt, cost = self.sess.run((self.optimizer, self.cost), 
                              feed_dict={self.x: X})
    return cost
  
  def transform(self, X):
    """Transform data by mapping it into the latent space."""
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run( [self.z_mean, self.z_log_sigma_sq],
                          feed_dict={self.x: X})
  
  def generate(self, z_mu=None):
    """ Generate data by sampling from latent space.
    
    If z_mu is not None, data for this point in latent space is
    generated. Otherwise, z_mu is drawn from prior in latent 
    space.
    """
    if z_mu is None:
      z_mu = np.random.normal(size=self.network_architecture["n_z"])
    # Note: This maps to mean of distribution, we could alternatively
    # sample from Gaussian distribution
    return self.sess.run(self.x_reconstr_mean, 
                         feed_dict={self.z: z_mu})
  
  def reconstruct(self, X):
    """ Use VAE to reconstruct given data. """
    return self.sess.run(self.x_reconstr_mean, 
                         feed_dict={self.x: X})


def train(vae,
          saver,
          batch_size,
          training_epochs,
          display_step=1):
  
  shapes = prepare_shapes()
  
  # Training cycle
  for epoch in range(training_epochs):
    # input用shape配列をシャッフルしておく
    random.shuffle(shapes)
    
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      # バッチを取得する.
      batch_xs = get_batch_images(shapes, batch_size, batch_size * i)
       # Fit training using batch data
      cost = vae.partial_fit(batch_xs)
      # Compute average loss
      avg_cost += cost / n_samples * batch_size
      
     # Display logs per epoch step
    if epoch % display_step == 0:
      print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    # checkpointへの保存
    saver.save(vae.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = epoch)

    
def reconstr_check(vae):
  # reconstructの確認
  x_sample = get_random_images(10)
  x_reconstruct = vae.reconstruct(x_sample)

  if not os.path.exists("reconstr_img"):
    os.mkdir("reconstr_img")

  for i in range(10):
    org_img      = x_sample[i].reshape(64, 64)
    reconstr_img = x_reconstruct[i].reshape(64, 64)
    imsave("reconstr_img/org_{0}.png".format(i),      org_img)
    imsave("reconstr_img/reconstr_{0}.png".format(i), reconstr_img)

    
def disentangle_check(vae, n_z):
  x = 0.2
  y = 0.2
  scale = 0.15
  rotate = 0.2
  img = generate_image(64, 64, x, y, scale, rotate)
  batch_xs = [img]
  z_mean, z_log_sigma_sq = vae.transform(batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  print(z_sigma_sq)
  z_m = z_mean[0]

  for target_z_index in range(10):
    for ri in range(10):
      value = -3.0 + (6.0 / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_m[i]
      reconstr_img = vae.generate(z_mu=z_mean2)
      rimg = reconstr_img[0].reshape(64, 64)
      imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)

def load_checkpoints():
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(vae.sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(CHECKPOINT_DIR):
      os.mkdir(CHECKPOINT_DIR)
  return saver


network_architecture = \
  dict(n_hidden_recog_1=1200, # 1st layer encoder neurons
       n_hidden_recog_2=1200, # 2nd layer encoder neurons
       n_hidden_gener_1=1200, # 1st layer decoder neurons
       n_hidden_gener_2=1200, # 2nd layer decoder neurons
       n_hidden_gener_3=1200, # 3rd layer decoder neurons
       n_input=4096, # data input (img shape: 64*64)
       n_z=10)       # dimensionality of latent space


vae = VariationalAutoencoder(network_architecture,
                             learning_rate=learning_rate,
                             batch_size=batch_size)

# Checkpointからの復元
saver = load_checkpoints()

# 学習
train(vae, saver, batch_size=batch_size, training_epochs=epoch_size)

# image reconstruction check
reconstr_check(vae)

# disentangle check
disentangle_check(vae, network_architecture["n_z"])
