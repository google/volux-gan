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

"""Library defining the modulated 1D convolutional layers."""

from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf


def get_weight_initializer_runtime_coef(
    shape: Sequence[int],
    gain: float = 1,
    use_wscale: bool = True,
    lrmul: float = 1.0) -> Tuple[float, float]:
  """Get initializer and lr coef for different weights shapes."""
  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
  fan_in = np.prod(shape[:-1])
  he_std = gain / np.sqrt(fan_in)  # He init

  # Equalized learning rate and custom learning rate multiplier.
  if use_wscale:
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
  else:
    init_std = he_std / lrmul
    runtime_coef = lrmul

  return init_std, runtime_coef


class Conv1DMod(tf.keras.layers.Layer):
  """Build a 1D convolutional layer with modulation operation."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int = 1,
               padding: str = 'SAME',
               demod: bool = True,
               device: str = 'gpu',
               **kwargs):
    """Initialize the 1D modulated convolutional layer.

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
    super(Conv1DMod, self).__init__(**kwargs)
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
      raise ValueError(
          'The channel dimension of the inputs should be defined. Found `None`.'
      )
    input_dim = input_shape[0][channel_axis]
    kernel_shape = (self._kernel_size,) + (input_dim, self._filters)

    if input_shape[1][-1] != input_dim:
      raise ValueError(
          'The last dimension of modulation input should be equal to input dimension.'
      )

    (self._init_std_w,
     self._runtime_coef_w) = get_weight_initializer_runtime_coef(
         shape=[self._kernel_size, input_dim, self._filters], lrmul=self._lrmul)

    self.kernel = self.add_weight(
        shape=kernel_shape,
        initializer=tf.random_normal_initializer(stddev=self._init_std_w),
        name='kernel')

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Run a forward pass of the convolutional layer."""
    feature, style_vector = inputs
    # Change style to (batch_size, 1, scales, 1)
    inp_mods = style_vector[:, None, :, None]

    # Kernel's weight is (kernel_size, input_maps, output_maps)
    my_kernel = self.kernel * self._runtime_coef_w

    # Modulate (scale) kernels [bs, kernel_size, input_maps, output_maps]
    weights = my_kernel[None, ...] * (inp_mods + 1)

    # Demodulate
    if self._demod:
      # Get variance by each output channel
      denominator = tf.math.rsqrt(
          tf.reduce_sum(tf.square(weights), axis=[1, 2],
                        keepdims=True))  # [BO] Scaling factor.
      weights = weights * denominator

    # Fuse kernels and fuse inputs
    feature = tf.reshape(feature, [1, -1, feature.shape[2]])  # [1, bs*ns, c]

    # [kernel_size, input_maps, bs, output_maps]
    weights = tf.transpose(weights, [1, 2, 0, 3])

    # [kernel_size, input_maps, batch_size*output_maps]
    weights = tf.reshape(weights, [weights.shape[0], weights.shape[1], -1])

    if self._device == 'gpu':
      feature = tf.nn.conv1d(
          feature,
          weights,
          stride=self._strides,
          padding=self._padding,
          data_format='NCW')
    else:
      feature = tf.transpose(feature, [0, 2, 1])
      feature = tf.nn.conv1d(
          feature,
          weights,
          stride=self._strides,
          padding=self._padding,
          data_format='NWC')
      feature = tf.transpose(feature, [0, 2, 1])

    # Un-fuse output
    # Fused => reshape convolution groups back to minibatch.
    feature = tf.reshape(feature, [-1, self._filters, tf.shape(feature)[2]])

    return feature
