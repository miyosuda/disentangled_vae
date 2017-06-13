# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vae import VariationalAutoencoder

class TestVariationalAutoencoder(tf.test.TestCase):
  def setUp(self):
    self.model = VariationalAutoencoder(0.01, 1.0)
  
  def get_batch_images(self, batch_size):
    image = np.zeros((4096), dtype=np.float32)
    batch_images = [image] * batch_size
    return batch_images
  
  def test_prepare_loss(self):
    batch_size = 10
    # loss variable should be scalar
    self.assertEqual( (), self.model.loss.get_shape() )
    
  def test_prepare_partial_fit(self):
    batch_xs = self.get_batch_images(10)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      # loss result should be float
      loss = self.model.partial_fit(sess, batch_xs)
      self.assertEqual(np.float32, loss.dtype)

  def test_transform(self):
    batch_xs = self.get_batch_images(10)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      z_mean, z_log_sigma_sq = self.model.transform(sess, batch_xs)
      # check shape of latent variables
      self.assertEqual( (10,10), z_mean.shape )
      self.assertEqual( (10,10), z_log_sigma_sq.shape )

  def test_generate(self):
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      # generate with prior
      xs = self.model.generate(sess)
      self.assertEqual( (1,4096), xs.shape )

      # generate with z_mu with batch size 5
      z_mu = np.zeros((5, 10), dtype=np.float32)
      xs = self.model.generate(sess, z_mu)
      self.assertEqual( (5,4096), xs.shape )

  def test_reconstruct(self):
    batch_xs = self.get_batch_images(10)    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      xs = self.model.reconstruct(sess, batch_xs)
      # check reconstructed image shape
      self.assertEqual( (10,4096), xs.shape )
      

if __name__ == "__main__":
  tf.test.main()
