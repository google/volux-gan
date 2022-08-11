#!/usr/bin/python
#
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library defining NeRF network and Upsampler."""
from typing import Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from generator.network import embedding_layers
from generator.network import modulated_conv1d as conv1_mod
from generator.network import modulated_conv2d as conv2_mod

_NUM_FREQS = 10


class BlurPool2D(tf.keras.layers.Layer):
  """Build 2D Blur Pooling Layer."""

  def __init__(self,
               pool_size: int = 2,
               kernel_size: int = 3,
               data_format: str = 'NCHW',
               **kwargs):
    """Initialize the 2D blur pooling layer.

    Args:
      pool_size: An integer specifying the channels of the hidden layers.
      kernel_size: An integer specifying the dimensions of the disentangled
        latent code for the MLPs of NeRF module.
      data_format: A string specifying data format of the input tensor, one of
        "NCHW" or "NHWC".
      **kwargs: Keyworded arguments that are forwarded by the model.
    """
    super(BlurPool2D, self).__init__(**kwargs)
    self.blur_kernel = None
    self.kernel_size = kernel_size
    self.data_format = data_format
    if data_format == 'NCHW':
      self.pool_size = (1, 1, pool_size, pool_size)
    elif data_format == 'NHWC':
      self.pool_size = (1, pool_size, pool_size, 1)

  def build(self, input_shape: tf.Tensor):
    """Build the model based on input shapes received."""
    if self.kernel_size == 3:
      blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
      blur_kernel = blur_kernel / np.sum(blur_kernel)
    elif self.kernel_size == 4:
      blur_kernel = np.array([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3],
                              [1, 3, 3, 1]])
      blur_kernel = blur_kernel / np.sum(blur_kernel)
    elif self.kernel_size == 5:
      blur_kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]])
      blur_kernel = blur_kernel / np.sum(blur_kernel)
    else:
      raise ValueError

    if self.data_format == 'NHWC':
      blur_kernel = np.repeat(blur_kernel, input_shape[3])
      blur_kernel = np.reshape(
          blur_kernel, (self.kernel_size, self.kernel_size, input_shape[3], 1))

      blur_init = tf.keras.initializers.constant(blur_kernel)

      self.blur_kernel = self.add_weight(
          name='blur_kernel',
          shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
          initializer=blur_init,
          trainable=False)
    else:
      blur_kernel = np.repeat(blur_kernel, input_shape[1])
      blur_kernel = np.reshape(
          blur_kernel, (self.kernel_size, self.kernel_size, input_shape[1], 1))

      blur_init = tf.keras.initializers.constant(blur_kernel)

      self.blur_kernel = self.add_weight(
          name='blur_kernel',
          shape=(self.kernel_size, self.kernel_size, input_shape[1], 1),
          initializer=blur_init,
          trainable=False)

  def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
    """Run a forward pass of the 2D bulr pooling."""
    output = tf.nn.depthwise_conv2d(
        input_tensor,
        self.blur_kernel,
        padding='SAME',
        data_format=self.data_format,
        strides=self.pool_size)

    return output


class DenseLayer(tf.keras.layers.Layer):
  """Dense layer with equalized learning rate."""

  def __init__(self, fmaps: int, lrmul: float = 1.0, **kwargs):
    """Initialize the dense layer."""
    super(DenseLayer, self).__init__(**kwargs)
    self._fmaps = fmaps
    self._lrmul = lrmul

  def build(self, input_shape: tf.Tensor):
    """Build the model based on input shapes received."""
    (init_std,
     self._runtime_coef) = conv1_mod.get_weight_initializer_runtime_coef(
         shape=[input_shape[-1], self._fmaps],
         gain=1,
         use_wscale=True,
         lrmul=self._lrmul)

    self.dense_weight = self.add_weight(
        name='weight',
        shape=(input_shape[-1], self._fmaps),
        initializer=tf.random_normal_initializer(stddev=init_std))
    self.dense_bias = self.add_weight(
        name='bias',
        shape=(self._fmaps,),
        initializer=tf.random_normal_initializer(stddev=init_std))

  def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
    """Run a forward pass of the dense layer."""
    input_tensor = tf.matmul(
        input_tensor, tf.math.multiply(self.dense_weight, self._runtime_coef))
    input_tensor = tf.nn.bias_add(
        input_tensor, self.dense_bias * self._lrmul, data_format='N...C')

    return input_tensor


class UniformBoxWarp(tf.keras.layers.Layer):
  """Scale the coordinates according the specified side length."""

  def __init__(self, sidelength: float):
    super().__init__()
    self._scale_factor = 2 / sidelength

  def call(self, coordinates: tf.Tensor) -> tf.Tensor:
    return coordinates * self._scale_factor


class CustomMappingNetwork(tf.keras.layers.Layer):
  """Build a Mapping network."""

  def __init__(self,
               hidden_dim: int = 512,
               nerf_dim: int = 256,
               conv_start_dim: int = 256,
               upsample_factor: int = 4,
               w_avg_beta: float = 0.995,
               style_mixing_prob: float = 0.9,
               num_freqs: int = _NUM_FREQS):
    """Initialize the Mapping Network.

    Args:
      hidden_dim: An integer specifying the channels of the hidden layers.
      nerf_dim: An integer specifying the dimensions of the disentangled latent
        code for the MLPs of NeRF module.
      conv_start_dim: An integer specifying the dimensions of the disentangled
        latent code for the MLPs of Upsampler.
      upsample_factor: An integer specifying the upsampling factor.
      w_avg_beta: A float specifying the decay for tracking the moving average
        of latent features during training.
      style_mixing_prob: A float specifying the probability of mixing two latent
        feature.
      num_freqs: An integer specifying the frequency number used for positional
        embedding.
    """
    super().__init__()
    self._initial_conv_dim = conv_start_dim
    self._level_num = int(np.log2(upsample_factor))
    self._mapping_layers = 8
    self._nerf_layers = 6
    self._lrmul = 0.01
    self._style_lrmul = 1.0
    self._w_avg_beta = w_avg_beta
    self._num_freqs = num_freqs

    self.w_avg = tf.Variable(
        name='w_avg',
        initial_value=tf.zeros([512]),
        dtype=tf.float32,
        trainable=False)

    self.network = []
    for _ in range(self._mapping_layers):
      self.network.append(DenseLayer(fmaps=hidden_dim, lrmul=self._lrmul))

    self.nerf_style_feature_layers = [
        DenseLayer(
            fmaps=3 * (self._num_freqs * 2 + 1), lrmul=self._style_lrmul)
    ]
    for _ in range(self._nerf_layers - 1):
      self.nerf_style_feature_layers.append(
          DenseLayer(fmaps=nerf_dim, lrmul=self._style_lrmul))

    self.conv_dims = []
    for level in range(self._level_num):
      self.conv_dims.append(conv_start_dim // 2**level)
      self.conv_dims.append(conv_start_dim // 2**(level + 1))
      self.conv_dims.append(conv_start_dim // 2**(level + 1))

    self.conv_style_feature_layers = []
    for conv_dim in self.conv_dims:
      self.conv_style_feature_layers.append(
          DenseLayer(fmaps=conv_dim, lrmul=self._style_lrmul))

    self.total_broadcast = self._nerf_layers + self._level_num * 3

    self.broadcast = tf.keras.layers.Lambda(
        lambda x: tf.tile(x[:, tf.newaxis], [1, self.total_broadcast, 1]))

  def call(
      self,
      x: tf.Tensor,
      update_average: bool = True
  ) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], tf.Tensor]:
    """Run a forward pass of the mapping network."""
    mapping_embeddings = self.inference_mapping_embeddings(x)
    mapping_embeddings = self.broadcast(mapping_embeddings)
    nerf_style_vectors, conv_style_vectors = self.inference_style_vectors(
        mapping_embeddings)

    if update_average:
      self.update_moving_average(mapping_embeddings)

    return (nerf_style_vectors, conv_style_vectors, mapping_embeddings)

  def inference_mapping_embeddings(self, x: tf.Tensor) -> tf.Tensor:
    """Inference latent feature from a random sampled latent code."""
    x = x * tf.math.rsqrt(
        tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)

    for layer in self.network:
      x = layer(x)
      x = tf.nn.leaky_relu(x, 0.2) * tf.math.sqrt(2.)

    return x

  def inference_style_vectors(
      self, x: tf.Tensor) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """Inference style features from a latent feature."""
    nerf_style_vectors = []
    conv_style_vectors = []

    x = tf.split(x, self.total_broadcast, axis=1)

    for layer in self.nerf_style_feature_layers:
      style_vector = layer(x[0][:, 0])
      x.pop(0)
      nerf_style_vectors.append(style_vector)

    for layer in self.conv_style_feature_layers:
      style_vector = layer(x[0][:, 0])
      x.pop(0)
      conv_style_vectors.append(style_vector)

    return (nerf_style_vectors, conv_style_vectors)

  def update_moving_average(self, w_latents: tf.Tensor) -> tf.Tensor:
    """Update moving average of latent feature."""
    batch_avg = tf.reduce_mean(w_latents[:, 0], axis=0)
    moved_w_avg = batch_avg + (self.w_avg - batch_avg) * self._w_avg_beta
    self.w_avg.assign(moved_w_avg)
    return w_latents


class NeRFBaseline(tf.keras.Model):
  """The NeRF module for generating neural implicit intrinsic field."""

  def __init__(self,
               hidden_dim: int = 256,
               num_freqs: int = _NUM_FREQS,
               device: str = 'gpu'):
    """Initialize NeRF module.

    Args:
      hidden_dim: An integer specifying the channels of the hidden layers.
      num_freqs: An integer specifying the frequency number used for positional
        embedding.
      device: A string specifying the type of device.
    """
    super().__init__()
    self.points_encoder = embedding_layers.SinusoidalEncoder(
        num_freqs, scale=1.0)
    self.bias_initializer = tf.random_normal_initializer(0, 0.1)
    self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.mod_convs = [
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
        conv1_mod.Conv1DMod(hidden_dim, kernel_size=1, device=device),
    ]

    self.bias = []
    bias = self.add_weight(
        name='Conv2DMod_bias_0',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)
    bias = self.add_weight(
        name='Conv2DMod_bias_1',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)
    bias = self.add_weight(
        name='Conv2DMod_bias_2',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)
    bias = self.add_weight(
        name='Conv2DMod_bias_3',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)
    bias = self.add_weight(
        name='Conv2DMod_bias_4',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)
    bias = self.add_weight(
        name='Conv2DMod_bias_5',
        shape=[hidden_dim],
        initializer=self.bias_initializer)
    self.bias.append(bias)

    self.final_layer = DenseLayer(fmaps=260)
    self.intrinsic_layer = tf.keras.Sequential()
    self.intrinsic_layer.add(DenseLayer(fmaps=32))
    self.intrinsic_layer.add(DenseLayer(fmaps=4))
    self.intrinsic_layer.add(tf.keras.layers.Activation('softmax'))

    self.gridwarper = UniformBoxWarp(0.25)

  def call(
      self,
      inputs: Tuple[tf.Tensor, Sequence[tf.Tensor]],
      return_normal: bool = True
  ) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Run a forward pass of the NeRF module."""
    (points, style_vectors) = inputs
    original_points = points
    with tf.GradientTape() as t:
      t.watch(original_points)

      points = self.gridwarper(points)
      points = self.points_encoder(points)
      points = tf.transpose(points, [0, 2, 1])

      for idx, mod_conv in enumerate(self.mod_convs):
        style_vector = style_vectors[idx]
        points = mod_conv([points, style_vector])
        points = tf.nn.bias_add(points, self.bias[idx], data_format='NCW')
        points = self.activation(points) * tf.math.sqrt(2.0)

      points = tf.transpose(points, [0, 2, 1])
      rgb_feature_alpha = self.final_layer(points)
      blend_weight = self.intrinsic_layer(points)
      opacity = rgb_feature_alpha[..., -1:]
      opacity = tf.nn.relu(opacity)

    rgb_weight_feature_alpha = tf.concat([
        tf.math.sigmoid(rgb_feature_alpha[..., :3]),
        blend_weight,
        rgb_feature_alpha[..., 3:],
    ],
                                         axis=-1)

    if return_normal:
      normals = t.gradient(opacity, original_points)
      normals = tf.math.l2_normalize(normals, axis=-1)
    else:
      normals = tf.zeros_like(original_points)

    return (rgb_weight_feature_alpha, normals)


class StyleConvNetwork(tf.keras.Model):
  """Build the Upsampling Module."""

  def __init__(
      self,
      start_dim: int = 256,
      upsample_factor: int = 4,
      device: str = 'gpu',
  ):
    """Initialize the Upsampling Network.

    Args:
      start_dim: An integer specifying the dimension of the input feature map.
      upsample_factor: An integer specifying the upsampling factor.
      device: A string specifying the type of device.
    """
    super().__init__()
    self._level_num = int(np.log2(upsample_factor))
    self.blur_pools = []
    self.mod_convs = []
    self.pixel_shuffle_mlps = []
    self.bias = []
    self.mlps_bias = []
    self.bias_initializer = tf.random_normal_initializer(0, 0.1)
    self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    self.to_rgb_high = []
    self.to_rgb_bias_high = []
    self._device = device

    for level_num in range(self._level_num):
      self.mod_convs.append(
          conv2_mod.Conv2DMod(
              start_dim // 2**(1 + level_num),
              kernel_size=1,
              padding='SAME',
              device=device))
      self.mod_convs.append(
          conv2_mod.Conv2DMod(
              start_dim // 2**(1 + level_num),
              kernel_size=1,
              padding='SAME',
              device=device))

      bias = self.add_weight(
          name='Conv2DMod_bias_%d_0' % level_num,
          shape=[start_dim // 2**(1 + level_num)],
          initializer=self.bias_initializer)
      self.bias.append(bias)
      bias = self.add_weight(
          name='Conv2DMod_bias_%d_1' % level_num,
          shape=[start_dim // 2**(1 + level_num)],
          initializer=self.bias_initializer)
      self.bias.append(bias)

      mlps = []
      mlps.append(DenseLayer(start_dim // 2**(1 + level_num) * 4))
      mlps.append(DenseLayer(start_dim // 2**(1 + level_num) * 4))
      self.pixel_shuffle_mlps.append(mlps)

      bias = self.add_weight(
          name='MLPs_bias_%d_0' % level_num,
          shape=[start_dim // 2**(1 + level_num) * 4],
          initializer=self.bias_initializer)
      self.mlps_bias.append(bias)
      bias = self.add_weight(
          name='MLPs_bias_%d_1' % level_num,
          shape=[start_dim // 2**(1 + level_num) * 4],
          initializer=self.bias_initializer)
      self.mlps_bias.append(bias)

      if self._device == 'gpu':
        self.blur_pools.append(BlurPool2D(pool_size=1, kernel_size=4))
      else:
        self.blur_pools.append(
            BlurPool2D(pool_size=1, kernel_size=4, data_format='NHWC'))

      self.to_rgb_high.append(
          conv2_mod.Conv2DMod(3, 1, demod=False, device=device))
      to_rgb_bias_high = self.add_weight(
          name='rgb_bias_high_%d' % level_num,
          shape=[3],
          initializer=self.bias_initializer)
      self.to_rgb_bias_high.append(to_rgb_bias_high)

  def call(
      self, inputs: Tuple[tf.Tensor, Sequence[tf.Tensor]]
  ) -> Tuple[Sequence[tf.Tensor], tf.Tensor]:
    """Run a forward pass of the upsampling module."""
    feature_map, style_vectors = inputs
    feature_map = tf.transpose(feature_map, [0, 3, 1, 2])
    rgb_maps = []

    for level_num in range(self._level_num):
      style_vector = style_vectors[level_num * 3]
      feature_map = self.mod_convs[level_num * 2]([feature_map, style_vector])

      add_feature_map = tf.transpose(feature_map, [0, 2, 3, 1])
      for index, layer in enumerate(self.pixel_shuffle_mlps[level_num]):
        add_feature_map = layer(add_feature_map)
        add_feature_map = tf.nn.bias_add(
            add_feature_map,
            self.mlps_bias[level_num * 2 + index],
            data_format='NHWC')
        if index == 0:
          add_feature_map = tf.nn.leaky_relu(add_feature_map, 0.2)
      add_feature_map = tf.transpose(add_feature_map, [0, 3, 1, 2])

      feature_map = tf.tile(feature_map, [1, 4, 1, 1]) + add_feature_map
      if self._device == 'gpu':
        feature_map = tf.nn.depth_to_space(feature_map, 2, data_format='NCHW')
        feature_map = self.blur_pools[level_num](feature_map)
      else:
        feature_map = tf.transpose(feature_map, [0, 2, 3, 1])
        feature_map = tf.nn.depth_to_space(feature_map, 2, data_format='NHWC')
        feature_map = self.blur_pools[level_num](feature_map)
        feature_map = tf.transpose(feature_map, [0, 3, 1, 2])
      feature_map = tf.nn.bias_add(
          feature_map, self.bias[level_num * 2], data_format='NCHW')
      feature_map = self.activation(feature_map) * tf.math.sqrt(2.0)
      style_vector = style_vectors[level_num * 3 + 1]
      feature_map = self.mod_convs[level_num * 2 +
                                   1]([feature_map, style_vector])
      feature_map = tf.nn.bias_add(
          feature_map, self.bias[level_num * 2 + 1], data_format='NCHW')
      feature_map = self.activation(feature_map) * tf.math.sqrt(2.0)

      style_vector = style_vectors[level_num * 3 + 2]
      high_rgb = self.to_rgb_high[level_num]([feature_map, style_vector])
      high_rgb = tf.nn.bias_add(
          high_rgb, self.to_rgb_bias_high[level_num], data_format='NCHW')
      high_rgb = tf.transpose(high_rgb, [0, 2, 3, 1])
      rgb_maps.append(high_rgb)

    feature_map = tf.transpose(feature_map, [0, 2, 3, 1])
    return rgb_maps, feature_map
