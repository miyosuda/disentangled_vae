import numpy as np
import tensorflow as tf

def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer



def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer




class BVAE(object):
    """ Based on Beta Variational Auto Encoder. """

    # Variacional
    def __init__(self,
                 gamma=100.0,
                 capacity_limit=25.0,
                 capacity_change_duration=100000,
                 learning_rate=5e-4):
        self.gamma = gamma
        self.capacity_limit = capacity_limit
        self.capacity_change_duration = capacity_change_duration
        self.learning_rate = learning_rate

        # Create autoencoder network
        self._create_network()

        # Define loss function and corresponding optimizer
        self._create_loss_optimizer()


    def _conv2d_weight_variable(self, weight_shape, name, deconv=False):
        with tf.name_scope(name):
            name_w = "W_{0}".format(name)
            name_b = "b_{0}".format(name)

            w = weight_shape[0]
            h = weight_shape[1]
            if deconv:
                input_channels  = weight_shape[3]
                output_channels = weight_shape[2]
            else:
                input_channels  = weight_shape[2]
                output_channels = weight_shape[3]
            d = 1.0 / np.sqrt(input_channels * w * h)
            bias_shape = [output_channels]

            weight = tf.get_variable(name_w, weight_shape,
                                     initializer=conv_initializer(w, h, input_channels))
            bias   = tf.get_variable(name_b, bias_shape,
                                     initializer=conv_initializer(w, h, input_channels))
            return weight, bias


    def _fc_weight_variable(self, weight_shape, name):
            name_w = "W_{0}".format(name)
            name_b = "b_{0}".format(name)

            input_channels  = weight_shape[0]
            output_channels = weight_shape[1]
            d = 1.0 / np.sqrt(input_channels)
            bias_shape = [output_channels]

            weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
            bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
            return weight, bias


    def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                  filter_width, row_stride, col_stride, padding_type):
        if padding_type == 'VALID':
            out_height = (input_height - 1) * row_stride + filter_height
            out_width  = (input_width  - 1) * col_stride + filter_width
        elif padding_type == 'SAME':
            out_height = input_height * row_stride
            out_width  = input_width * col_stride
        return out_height, out_width


    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                            padding='SAME')


    def _deconv2d(self, x, W, input_width, input_height, stride):
        filter_height = W.get_shape()[0].value
        filter_width  = W.get_shape()[1].value
        out_channel   = W.get_shape()[2].value

        out_height, out_width = self._get_deconv2d_output_size(input_height,
                                                               input_width,
                                                               filter_height,
                                                               filter_width,
                                                               stride,
                                                               stride,
                                                               'SAME')
        batch_size = tf.shape(x)[0]
        output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
        return tf.nn.conv2d_transpose(x, W, output_shape,
                                      strides=[1, stride, stride, 1],
                                      padding='SAME')


    def _sample_z(self, z_mean, z_log_sigma_sq):
        with tf.variable_scope("sample_Z") as scope:
            eps_shape = tf.shape(z_mean)
            eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
            # z = mu + sigma * epsilon
            z = tf.add(z_mean,
                       tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
            return z


    def _create_encoder_network(self, x, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            # [filter_height, filter_width, in_channels, out_channels]
            #with tf.variable_scope("c1")

            x_reshaped = tf.reshape(x, [-1, 64, 64, 1])

            with tf.variable_scope("conv1"):
                W_conv1, b_conv1 = self._conv2d_weight_variable([4, 4, 1,  32], "conv1")
                h_conv1 = tf.nn.relu(self._conv2d(x_reshaped, W_conv1, 2) + b_conv1)  # (32, 32)

            with tf.variable_scope("conv2"):
                W_conv2, b_conv2 = self._conv2d_weight_variable([4, 4, 32, 32], "conv2")
                h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 2) + b_conv2)  # (16, 16)

            with tf.variable_scope("conv3"):
                W_conv3, b_conv3 = self._conv2d_weight_variable([4, 4, 32, 32], "conv3")
                h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 2) + b_conv3)  # (8, 8)

            with tf.variable_scope("conv4"):
                W_conv4, b_conv4 = self._conv2d_weight_variable([4, 4, 32, 32], "conv4")
                h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4, 2) + b_conv4)  # (4, 4)

            with tf.variable_scope("fc1"):
                h_conv4_flat = tf.reshape(h_conv4, [-1, 4* 4 * 32])
                W_fc1, b_fc1 = self._fc_weight_variable([4 * 4 * 32, 256], "fc1")
                h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)

            with tf.variable_scope("fc2"):
                W_fc2, b_fc2 = self._fc_weight_variable([256, 256], "fc2")
                h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            return (h_fc2)


    def _create_z_network(self, h_fc2, reuse=False):
        with tf.variable_scope("latent", reuse=reuse) as scope:
            W_fc3, b_fc3 = self._fc_weight_variable([256, 10], "z_mean")
            W_fc4, b_fc4 = self._fc_weight_variable([256, 10], "z_sigma")
            z_mean = tf.matmul(h_fc2, W_fc3) + b_fc3
            z_log_sigma_sq = tf.matmul(h_fc2, W_fc4) + b_fc4
            z_mean_op=tf.summary.histogram("z_mean", z_mean)
            z_sigma_op=tf.summary.histogram("z_sigma", z_log_sigma_sq)

        return (z_mean,z_log_sigma_sq)


    def _create_decoder_network(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse) as scope:

            with tf.variable_scope("fc1") as scope:
                W_fc1, b_fc1 = self._fc_weight_variable([10, 256], "fc1")
                h_fc1 = tf.nn.relu(tf.matmul(z, W_fc1) + b_fc1)

            with tf.variable_scope("fc2") as scope:
                W_fc2, b_fc2 = self._fc_weight_variable([256, 4 * 4 * 32], "fc2")
                h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
                h_fc2_reshaped = tf.reshape(h_fc2, [-1, 4, 4, 32])

            with tf.variable_scope("deconv1") as scope:
                # [filter_height, filter_width, output_channels, in_channels]
                W_deconv1, b_deconv1 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv1", deconv=True)
                h_deconv1 = tf.nn.relu(self._deconv2d(h_fc2_reshaped, W_deconv1, 4, 4, 2) + b_deconv1)

            with tf.variable_scope("deconv2") as scope:
                W_deconv2, b_deconv2 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv2", deconv=True)
                h_deconv2 = tf.nn.relu(self._deconv2d(h_deconv1, W_deconv2, 8, 8, 2) + b_deconv2)

            with tf.variable_scope("deconv3") as scope:
                W_deconv3, b_deconv3 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
                h_deconv3 = tf.nn.relu(self._deconv2d(h_deconv2, W_deconv3, 16, 16, 2) + b_deconv3)

            with tf.variable_scope("deconv4") as scope:
                W_deconv4, b_deconv4 = self._conv2d_weight_variable([4, 4, 1, 32], "deconv4", deconv=True)
                h_deconv4 = self._deconv2d(h_deconv3, W_deconv4, 32, 32, 2) + b_deconv4
                x_out_logit = tf.reshape(h_deconv4, [-1, 64 * 64 * 1])

            return x_out_logit


    def _create_network(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, shape=[None, 4096], name="x")

        with tf.variable_scope("B-Vae"):

            self.h_fc2 = self._create_encoder_network(self.x)
            self.z_mean, self.z_log_sigma_sq = self._create_z_network(self.h_fc2)
            # Draw one sample z from Gaussian distribution
            # z = mu + sigma * epsilon
            self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
            self.x_out_logit = self._create_decoder_network(self.z)
            self.x_out = tf.nn.sigmoid(self.x_out_logit)


    def _create_loss_optimizer(self):
        with tf.variable_scope("loss"):
            with tf.variable_scope("reconstruction_loss"):
                # Reconstruction loss
                reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                        logits=self.x_out_logit)
                reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
                self.reconstr_loss = tf.reduce_mean(reconstr_loss)
                reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)

            with tf.variable_scope("latent_loss"):
            # Latent loss
                latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                   - tf.square(self.z_mean)
                                                   - tf.exp(self.z_log_sigma_sq), 1)
                self.latent_loss = tf.reduce_mean(latent_loss)
                latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)

            # Encoding capacity
            self.capacity = tf.placeholder(tf.float32, shape=[], name="capacity")
            # Loss with encoding capacity term
            self.loss = self.reconstr_loss + self.gamma * tf.abs(self.latent_loss - self.capacity)

            self.summary_op = tf.summary.merge_all()

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)


    def _calc_encoding_capacity(self, step):
        if step > self.capacity_change_duration:
            c = self.capacity_limit
        else:
            c = self.capacity_limit * (step / self.capacity_change_duration)
        return c


    def partial_fit(self, sess, xs, step):
        """Train model based on mini-batch of input data.

        Return loss of mini-batch.
        """
        c = self._calc_encoding_capacity(step)
        _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                               self.reconstr_loss,
                                                               self.latent_loss,
                                                               self.summary_op),
                                                              feed_dict={
                                                                  self.x: xs,
                                                                  self.capacity: c
                                                              })
        return reconstr_loss, latent_loss, summary_str


    def reconstruct(self, sess, xs):
        """ Reconstruct given data. """
        # Original VAE output
        return sess.run(self.x_out,
                        feed_dict={self.x: xs})


    def transform(self, sess, xs):
        """Transform data by mapping it into the latent space."""
        return sess.run([self.z_mean, self.z_log_sigma_sq],
                        feed_dict={self.x: xs})


    def generate(self, sess, zs):
        """ Generate data by sampling from latent space. """
        return sess.run(self.x_out,
                        feed_dict={self.z: zs})
