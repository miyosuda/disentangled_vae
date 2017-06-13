# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# this VAE implementation is based on Jan Hendrik Metzen's code
# https://jmetzen.github.io/2015-11-27/vae.html

import numpy as np
import tensorflow as tf

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
  """ Variation Autoencoder """
  def __init__(self,
               learning_rate,
               beta):
    self._create_network()
    self._prepare_loss(learning_rate, beta)

  def _create_network(self):
    n_hidden_encode_1 = 1200
    n_hidden_encode_2 = 1200
    n_hidden_decode_1 = 1200
    n_hidden_decode_2 = 1200
    n_hidden_decode_3 = 1200
    n_input           = 4096
    n_z               = 10
    
    network_weights = self._initialize_weights(n_hidden_encode_1,
                                               n_hidden_encode_2,
                                               n_hidden_decode_1,
                                               n_hidden_decode_2,
                                               n_hidden_decode_3,
                                               n_input,
                                               n_z)

    # tf Graph input (batch x 4096)
    self.x = tf.placeholder(tf.float32, [None, 4096])
    
    self.z_mean, self.z_log_sigma_sq = \
      self._create_encoder_network(network_weights["weights_encode"], 
                                   network_weights["biases_encode"])

    batch_sz = tf.shape(self.x)[0]
    eps_shape = tf.stack([batch_sz, n_z])

    # mean=0.0, stddev=1.0    
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    
    # z = mu + sigma * epsilon
    self.z = tf.add(self.z_mean, 
                    tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    self.x_reconstr_mean_logit, self.x_reconstr_mean = \
      self._create_decoder_network(network_weights["weights_decode"],
                                   network_weights["biases_decode"])
    
  def _initialize_weights(self,
                          n_hidden_encode_1, # 1200
                          n_hidden_encode_2, # 1200
                          n_hidden_decode_1, # 1200
                          n_hidden_decode_2, # 1200
                          n_hidden_decode_3, # 1200
                          n_input,           # 4096
                          n_z):              # 10
    all_weights = dict()
    
    all_weights['weights_encode'] = {
      'h1'           : tf.Variable(xavier_init(n_input,           n_hidden_encode_1)),
      'h2'           : tf.Variable(xavier_init(n_hidden_encode_1, n_hidden_encode_2)),
      'out_mean'     : tf.Variable(xavier_init(n_hidden_encode_2, n_z)),
      'out_log_sigma': tf.Variable(xavier_init(n_hidden_encode_2, n_z))}
    
    all_weights['biases_encode'] = {
      'b1'           : tf.Variable(tf.zeros([n_hidden_encode_1], dtype=tf.float32)),
      'b2'           : tf.Variable(tf.zeros([n_hidden_encode_2], dtype=tf.float32)),
      'out_mean'     : tf.Variable(tf.zeros([n_z],               dtype=tf.float32)),
      'out_log_sigma': tf.Variable(tf.zeros([n_z],               dtype=tf.float32))}
    
    all_weights['weights_decode'] = {
      'h1'           : tf.Variable(xavier_init(n_z,               n_hidden_decode_1)),
      'h2'           : tf.Variable(xavier_init(n_hidden_decode_1, n_hidden_decode_2)),
      'h3'           : tf.Variable(xavier_init(n_hidden_decode_2, n_hidden_decode_3)),
      'out_mean'     : tf.Variable(xavier_init(n_hidden_decode_3, n_input)),
      'out_log_sigma': tf.Variable(xavier_init(n_hidden_decode_3, n_input))}
    
    all_weights['biases_decode'] = {
      'b1'           : tf.Variable(tf.zeros([n_hidden_decode_1], dtype=tf.float32)),
      'b2'           : tf.Variable(tf.zeros([n_hidden_decode_2], dtype=tf.float32)),
      'b3'           : tf.Variable(tf.zeros([n_hidden_decode_3], dtype=tf.float32)),
      'out_mean'     : tf.Variable(tf.zeros([n_input],           dtype=tf.float32)),
      'out_log_sigma': tf.Variable(tf.zeros([n_input],           dtype=tf.float32))}
    
    return all_weights
  
  def _create_encoder_network(self, weights, biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x,  weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    z_mean         = tf.add(tf.matmul(layer_2, weights['out_mean']),
                            biases['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                            biases['out_log_sigma'])
    return (z_mean, z_log_sigma_sq)

  def _create_decoder_network(self, weights, biases):
    layer_1 = tf.tanh(tf.add(tf.matmul(self.z,  weights['h1']), biases['b1']))
    layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.tanh(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    x_reconstr_mean_logit = tf.add(tf.matmul(layer_3, weights['out_mean']),
                                   biases['out_mean'])
    x_reconstr_mean = tf.nn.sigmoid(x_reconstr_mean_logit)
    return x_reconstr_mean_logit, x_reconstr_mean
      
  def _prepare_loss(self, learning_rate, beta):
    # reconstruction loss (the negative log probability)
    reconstr_loss = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                              logits=self.x_reconstr_mean_logit),
      1)
    
    # latent loss
    latent_loss = beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                              - tf.square(self.z_mean)
                                              - tf.exp(self.z_log_sigma_sq), 1)

    # average over batch
    self.loss = tf.reduce_mean(reconstr_loss + latent_loss)
    self.optimizer = tf.train.AdagradOptimizer(
      learning_rate=learning_rate).minimize(self.loss)
    
  def partial_fit(self, sess, X, summary_op=None):
    """Train model based on mini-batch of input data.
    Return loss of mini-batch.
    """
    if summary_op is None:
      _, loss = sess.run( (self.optimizer, self.loss), 
                          feed_dict={self.x: X} )
      return loss
    else:
      _, loss, summary_str = sess.run( (self.optimizer, self.loss, summary_op),
                                       feed_dict={self.x: X} )
      return loss, summary_str
      
  
  def transform(self, sess, X):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: X} )
  
  def generate(self, sess, z_mu=None):
    """ Generate data by sampling from latent space. """
    if z_mu is None:
      z_mu = np.random.normal(size=(1,10))
    return sess.run( self.x_reconstr_mean, 
                     feed_dict={self.z: z_mu} )
  
  def reconstruct(self, sess, X):
    """ Use VAE to reconstruct given data. """
    return sess.run( self.x_reconstr_mean, 
                     feed_dict={self.x: X} )
