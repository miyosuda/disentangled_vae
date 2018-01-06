# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import VAE



class VAETest(tf.test.TestCase):
  def get_batch_images(self, batch_size):
    image = np.zeros((4096), dtype=np.float32)
    batch_images = [image] * batch_size
    return batch_images

  def test_prepare_loss(self):
    model = VAE()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      
      self.assertEqual(model.loss.get_shape(), ())
      self.assertEqual(model.x_out_logit.get_shape()[1], 64*64)
      self.assertEqual(model.x_out.get_shape()[1],       64*64)

  def test_partial_fit(self):
    model = VAE()
    
    batch_size = 10
    batch_xs = self.get_batch_images(batch_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      
      # loss result should be float
      reconstr_loss, latent_loss, _ = model.partial_fit(sess, batch_xs, 0)
      self.assertEqual(np.float32, reconstr_loss.dtype)
      self.assertEqual(np.float32, latent_loss.dtype)

  def test_transform(self):
    model = VAE()
    
    batch_size = 10
    batch_xs = self.get_batch_images(batch_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
      # check shape of latent variables
      self.assertEqual( (batch_size, 10), z_mean.shape )
      self.assertEqual( (batch_size, 10), z_log_sigma_sq.shape )

  def test_generate(self):
    model = VAE()
    
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      # generate with z_mu with batch size 5
      batch_size = 5
      z_mu = np.zeros((batch_size, 10), dtype=np.float32)
      xs = model.generate(sess, z_mu)
      self.assertEqual( (batch_size, 4096), xs.shape )

  def test_reconstruct(self):
    batch_size = 10
    batch_xs = self.get_batch_images(batch_size)

    model = VAE()
      
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
        
      xs = model.reconstruct(sess, batch_xs)
      # check reconstructed image shape
      self.assertEqual( (batch_size, 4096), xs.shape )

    

if __name__ == "__main__":
  tf.test.main()
