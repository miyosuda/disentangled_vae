# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import gc

from data_manager import DataManager


class DataManagerTest(unittest.TestCase):
  def setUp(self):
    self.manager = DataManager()
    self.manager.load()

  def tearDown(self):
    del self.manager
    gc.collect()
  
  def test_load(self):
    # sample size should be 737280
    self.assertEquals(self.manager.sample_size, 737280)

  def test_get_image(self):
    # get first image
    image0 = self.manager.get_image(shape=0, scale=0, orientation=0, x=0, y=0)
    self.assertTrue(image0.shape == (4096,))
    
    # boundary check
    image1 = self.manager.get_image(shape=3-1, scale=6-1, orientation=40-1,
                                    x=32-1, y=32-1)
    self.assertTrue(image1.shape == (4096,))

  def test_get_images(self):
    indices = [0,1,2]
    images = self.manager.get_images(indices)
    
    self.assertEquals(len(images), 3)
    # image shpe should be flatten. (4096)
    self.assertTrue(images[0].shape == (4096,))

  def test_get_random_images(self):
    images = self.manager.get_random_images(3)
    
    self.assertEquals(len(images), 3)
    self.assertTrue(images[0].shape == (4096,))

if __name__ == '__main__':
  unittest.main()
