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

"""Library defining the modulated convolutional layers."""
from typing import Optional, Tuple

import tensorflow as tf

from generator.network import modulated_conv1d
from generator.network import upfirdn2d


def modulated_conv2d(
    features: tf.Tensor,
    weight: tf.Tensor,
    styles: tf.Tensor,
    noise: Optional[tf.Tensor] = None,
    up: int = 1,
    down: int = 1,
    resample_filter: Optional[tf.Tensor] = None,
    demodulate: bool = True,
    fused_modconv: bool = True,
    pre_normalize: bool = False,
    device: str = 'gpu',
) -> tf.Tensor:
  """Apply 2D convolution with modulation.

  Args:
    features: A Tensor of size B x C x H x W, where B is batch size, H x W is
      the image size, and C is the number of channels.
    weight: A Tensor of size H x W x C_in x C_out, H x W is the kernel size,
      C_in is the number of in channels, , C_out is the number of out channels.
    styles: A Tensor of size B x C_in, where B is batch size, C_in is the number
      of in channels.
    noise: A Tensor of size B x C_out x H x W, where B is batch size, H x W is
      the image size, and C_out is the number of out channels, or None.
    up: A integer upsampling factor.
    down: A integer downsampling factor.
    resample_filter: A Tensor of size H x W, H x W is the filter size.
    demodulate: A bool indicating whether to apply weight demodulation.
    fused_modconv: A bool indicating whether to Perform modulation, convolution,
      and demodulation as a single fused operation.
    pre_normalize: A bool indicating whether to normalize weight and styles.
    device: A string specifying the type of device.

  Returns:
    outputs: The features of size B x H*Up/Down x W*Up/Down x C_out.
  """
  out_channels = weight.shape.as_list()[3]

  if pre_normalize:
    weight = weight * tf.math.rsqrt(
        tf.reduce_mean(tf.square(weight), axis=[0, 1, 2], keepdims=True) + 1e-8)
    styles = styles * tf.reduce_max(styles, axis=1, keepdims=True)

  # Calculate per-sample weights and demodulation coefficients.
  weight = weight[None, ...]  # [BKKIO]
  weight = weight * (styles[:, None, None, :, None] + 1)  # [BKKIO]
  if demodulate:
    dcoefs = tf.math.rsqrt(
        tf.reduce_sum(tf.square(weight), axis=[1, 2, 3], keepdims=False) +
        1e-8)  # [B111O]
    weight = weight * dcoefs[:, None, None, None, :]  # [BKKIO]

  if fused_modconv:
    # Execute as one fused op using grouped convolution.
    features = tf.reshape(
        features,
        [1, -1, features.shape[2], features.shape[3]])  # [1, if*bs, h, w]

    # [3, 3, input_maps, bs, output_maps]
    weight = tf.transpose(weight, [1, 2, 3, 0, 4])
    # [3, 3, input_maps, batch_size*output_maps]
    weight = tf.reshape(weight,
                        [weight.shape[0], weight.shape[1], weight.shape[2], -1])
  else:
    features *= styles[:, :, None, None]

  if up > 1:
    features = upsample_conv_2d(
        features, weight, resample_filter, factor=up, device=device)
  elif down > 1:
    features = conv_downsample_2d(
        features, weight, resample_filter, factor=down, device=device)
  else:
    if device == 'gpu':
      features = tf.nn.conv2d(
          features,
          weight,
          data_format='NCHW',
          strides=[1, 1, 1, 1],
          padding='SAME')
    else:
      features = tf.transpose(features, [0, 2, 3, 1])
      features = tf.nn.conv2d(
          features,
          weight,
          strides=[1, 1, 1, 1],
          padding='VALID',
          data_format='NHWC')
      features = tf.transpose(features, [0, 3, 1, 2])

  # Un-fuse output
  if fused_modconv:
    features = tf.reshape(
        features, [-1, out_channels, features.shape[2], features.shape[3]])
  elif demodulate:
    features *= dcoefs[:, :None, None]  # [BKKIO]

  if noise is not None:
    features = features + noise
  return features


def upsample_conv_2d(features: tf.Tensor,
                     weight: tf.Tensor,
                     resample_filter: tf.Tensor,
                     factor: int = 2,
                     gain: float = 1.0,
                     device: str = 'gpu') -> tf.Tensor:
  """Apply 2D convolution with upsampling operation.

  Args:
    features: A Tensor of size B x C x H x W, where B is batch size, H x W is
      the image size, and C is the number of channels.
    weight: A Tensor of size H x W x C_in x C_out, H x W is the kernel size,
      C_in is the number of in channels, , C_out is the number of out channels.
    resample_filter: A Tensor of size H x W, H x W is the filter size.
    factor: A int indicating the upsampling factor.
    gain: A float indicating the gain factor.
    device: A string specifying the type of device.

  Returns:
    outputs: The features of size B x H*factor x W*factor x C_out.
  """
  batch_size, channels, height, width = features.shape.as_list()
  kernel_size = weight.shape.as_list()[0]
  in_channels, out_channels = weight.shape.as_list()[2:4]
  filter_size = resample_filter.shape.as_list()[0]

  resample_filter = resample_filter * (gain * (factor**2))
  padding = (filter_size - factor) - (kernel_size - 1)

  stride = factor
  num_groups = channels // in_channels

  weight = tf.reshape(weight,
                      [kernel_size, kernel_size, in_channels, num_groups, -1])
  weight = tf.transpose(weight, [0, 1, 4, 3, 2])
  weight = tf.reshape(weight,
                      [kernel_size, kernel_size, -1, num_groups * in_channels])
  if device == 'gpu':
    output_shape = [
        batch_size, out_channels, (height - 1) * factor + kernel_size,
        (width - 1) * factor + kernel_size
    ]
    features = tf.nn.conv2d_transpose(
        features,
        weight,
        output_shape=output_shape,
        strides=stride,
        padding='VALID',
        data_format='NCHW')
  else:
    output_shape = [
        batch_size, (height - 1) * factor + kernel_size,
        (width - 1) * factor + kernel_size, out_channels
    ]
    features = tf.transpose(features, [0, 2, 3, 1])
    features = tf.nn.conv2d_transpose(
        features,
        weight,
        output_shape=output_shape,
        strides=stride,
        padding='VALID',
        data_format='NHWC')
    features = tf.transpose(features, [0, 3, 1, 2])

  return upfirdn2d.simple_upfirdn_2d(
      features,
      resample_filter,
      pad0=(padding + 1) // 2 + factor - 1,
      pad1=padding // 2 + 1,
      device=device)


def conv_downsample_2d(features: tf.Tensor,
                       weight: tf.Tensor,
                       resample_filter: tf.Tensor,
                       factor: int = 2,
                       gain: float = 1,
                       device: str = 'gpu') -> tf.Tensor:
  """Apply 2D convolution with downsampling operation.

  Args:
    features: A Tensor of size B x C x H x W, where B is batch size, H x W is
      the image size, and C is the number of channels.
    weight: A Tensor of size H x W x C_in x C_out, H x W is the kernel size,
      C_in is the number of in channels, , C_out is the number of out channels.
    resample_filter: A Tensor of size H x W, H x W is the filter size.
    factor: A int indicating the upsampling factor.
    gain: A float indicating the gain factor.
    device: A string specifying the type of device.

  Returns:
    outputs: The features of size B x H/factor x W/factor x C_out.
  """
  kernel_size = weight.shape.as_list()[0]
  filter_size = resample_filter.shape.as_list()[0]

  resample_filter = resample_filter * gain
  padding = (filter_size - factor) + (kernel_size - 1)

  stride = factor
  features = upfirdn2d.simple_upfirdn_2d(
      features,
      resample_filter,
      pad0=(padding + 1) // 2,
      pad1=padding // 2,
      device=device)

  if device == 'gpu':
    features = tf.nn.conv2d(
        features, weight, strides=stride, padding='VALID', data_format='NCHW')
  else:
    features = tf.transpose(features, [0, 2, 3, 1])
    features = tf.nn.conv2d(
        features, weight, strides=stride, padding='VALID', data_format='NHWC')
    features = tf.transpose(features, [0, 3, 1, 2])

  return features


class Conv2DMod(tf.keras.layers.Layer):
  """Build a 2D convolutional layer with modulation operation."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int = 1,
               padding: str = 'SAME',
               demod: bool = True,
               device: str = 'gpu',
               **kwargs):
    """Initialize the 2D modulated convolutional layer.

    Args:
      filters: An integer indicating the dimensionality of the output space.
      kernel_size: An integer specifying the length of the conv window.
      strides: An integer specifying the stride length of the convolution.
      padding: A string specifying padding mode, one of "VALID" or "SAME".
      demod: A bool indicating whether to apply weight demodulation.
      device: A string specifying the type of device.
      **kwargs: Keyworded arguments that are forwarded by the model.

    Raises:
      ValueError:
        - 'The channel dimension of the inputs should be defined. Found `None`.'
        - 'The last dimension of modulation input should be equal to input
        dimension.'
    """
    super(Conv2DMod, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._demod = demod
    self._lrmul = 1.0
    self._device = device

  def build(self, input_shape: Tuple[tf.Tensor, tf.Tensor]):
    """Builds the model based on input shapes received."""
    channel_axis = 1
    if input_shape[0][channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[0][channel_axis]
    kernel_shape = (self._kernel_size, self._kernel_size) + (input_dim,
                                                             self._filters)

    if input_shape[1][-1] != input_dim:
      raise ValueError('The last dimension of modulation input should '
                       'be equal to input dimension.')

    (self._init_std_w, self._runtime_coef_w
    ) = modulated_conv1d.get_weight_initializer_runtime_coef(
        shape=[self._kernel_size, self._kernel_size, input_dim, self._filters],
        lrmul=self._lrmul)

    self.kernel = self.add_weight(
        shape=kernel_shape,
        initializer=tf.random_normal_initializer(stddev=self._init_std_w),
        name='kernel')

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Run a forward pass of the convolutional layer."""
    feature, style_vector = inputs
    # Make w's shape compatible with self.kernel
    # Kernel's weight is (3, 3, input_maps, output_maps)
    # Change style vector to (batch_size, 1, 1, scales, 1)
    inp_mods = style_vector[:, None, None, :, None]

    # Add minibatch layer to weights: (1, 3, 3, input_maps, output_maps)
    my_kernel = self.kernel * self._runtime_coef_w

    # Modulate (scale) kernels [bs, 3, 3, input_maps, output_maps]
    weights = my_kernel[None, ...] * (inp_mods + 1)

    # Demodulate
    if self._demod:
      # Get variance by each output channel
      denominator = tf.math.rsqrt(
          tf.reduce_sum(tf.square(weights), axis=[1, 2, 3],
                        keepdims=True))  # [BO] Scaling factor.
      weights = weights * denominator

    # Fuse kernels and fuse inputs
    feature = tf.reshape(
        feature,
        [1, -1, feature.shape[2], feature.shape[3]])  # [1, if*bs, h, w]

    # [3, 3, input_maps, bs, output_maps]
    weights = tf.transpose(weights, [1, 2, 3, 0, 4])
    # [3, 3, input_maps, output_maps*batch_size]
    weights = tf.reshape(
        weights, [weights.shape[0], weights.shape[1], weights.shape[2], -1])

    if self._device == 'gpu':
      feature = tf.nn.conv2d(
          feature,
          weights,
          strides=self._strides,
          padding=self._padding,
          data_format='NCHW')
    else:
      feature = tf.transpose(feature, [0, 2, 3, 1])
      feature = tf.nn.conv2d(
          feature,
          weights,
          strides=self._strides,
          padding=self._padding,
          data_format='NHWC')
      feature = tf.transpose(feature, [0, 3, 1, 2])

    # Un-fuse output
    feature = tf.reshape(
        feature,
        [-1, self._filters,
         tf.shape(feature)[2],
         tf.shape(feature)[3]])
    # Fused => reshape convolution groups back to minibatch.

    return feature


class Conv2D(tf.keras.layers.Layer):
  """Build the 2D convolutional layer with equalized learning rate."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int = 1,
               padding: str = 'SAME',
               **kwargs):
    """Initialize the 2D convolutional layer with equalized learning rate.

    Args:
      filters: An integer indicating the dimensionality of the output space.
      kernel_size: An integer specifying the length of the conv window.
      strides: An integer specifying the stride length of the convolution.
      padding: A string specifying padding mode, one of "VALID" or "SAME".
      **kwargs: Keyworded arguments that are forwarded by the model.

    Raises:
      ValueError:
        - 'The channel dimension of the inputs should be defined. Found `None`.'
    """
    super(Conv2D, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._lrmul = 1.0

  def build(self, input_shape: tf.Tensor):
    """Builds the model based on input shapes received."""
    channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = (self._kernel_size, self._kernel_size) + (input_dim,
                                                             self._filters)

    (self._init_std_w, self._runtime_coef_w
    ) = modulated_conv1d.get_weight_initializer_runtime_coef(
        shape=[self._kernel_size, self._kernel_size, input_dim, self._filters],
        lrmul=self._lrmul)

    self.kernel = self.add_weight(
        shape=kernel_shape,
        initializer=tf.random_normal_initializer(stddev=self._init_std_w),
        name='kernel')

    self.bias = self.add_weight(
        name='bias',
        shape=(self._filters,),
        initializer=tf.random_normal_initializer(stddev=self._init_std_w))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Run a forward pass of the convolutional layer."""
    kernel = self.kernel * self._runtime_coef_w

    x = tf.nn.conv2d(
        x,
        kernel,
        strides=self._strides,
        padding=self._padding,
        data_format='NHWC')
    x = tf.nn.bias_add(x, self.bias * self._lrmul, data_format='N...C')

    return x
