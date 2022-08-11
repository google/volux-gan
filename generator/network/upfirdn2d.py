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

"""Custom Tensorflow ops for efficient resampling of 2D images."""

from typing import Sequence

import numpy as np
import tensorflow as tf


def setup_filter(filters: Sequence[int],
                 normalize: bool = True,
                 flip_filter: bool = False,
                 gain: float = 1,
                 separable: bool = False) -> tf.Tensor:
  """Convenience function to setup 2D FIR filter for `upfirdn2d()`.

  Args:
    filters: A numpy array, or python list of the shape `[filter_height,
      filter_width]` (non-separable) or `[filter_taps]` (separable) to represent
      the sampling weight, e.g., [1,3,3,1]
    normalize: Normalize the filter so that it retains the magnitude for
      constant input signal (DC)? (default: True).
    flip_filter: Flip the filter? (default: False).
    gain: Overall scaling factor for signal magnitude (default: 1).
    separable:   Return a separable filter? (default: select automatically).

  Returns:
      Float32 tensor of the shape `[filter_height, filter_width]`
  """
  # Validate.
  if filters is None:
    filters = 1
  filters = np.asarray(filters, dtype=float)

  assert filters.ndim in [0, 1, 2]
  if filters.ndim == 0:
    filters = filters[np.newaxis]

  # Separable?
  if not separable and filters.ndim == 1:
    filters = filters[:, np.newaxis] * filters[np.newaxis, :]

  # Apply normalize, flip and gain.
  if normalize:
    filters /= filters.sum()
  if flip_filter:
    filters = filters.flip(list(range(filters.ndim)))

  filters = filters * (gain**(filters.ndim / 2))
  filters = tf.constant(filters, dtype=tf.float32)
  return filters


def simple_upfirdn_2d(features: tf.Tensor,
                      filters: tf.Tensor,
                      up: int = 1,
                      down: int = 1,
                      pad0: int = 0,
                      pad1: int = 0,
                      device: str = 'gpu') -> tf.Tensor:
  """The generic API for upfirdn_2d() function."""
  features = upfirdn_2d(
      features,
      filters,
      upx=up,
      upy=up,
      downx=down,
      downy=down,
      padx0=pad0,
      padx1=pad1,
      pady0=pad0,
      pady1=pad1,
      device=device)

  return features


def upsample2d(image: tf.Tensor,
               filters: tf.Tensor,
               up_scale: int = 2,
               gain: int = 1,
               device: str = 'gpu') -> tf.Tensor:
  """Apply upfirdn_2d() and upsample the image."""
  filters = filters * (gain * (up_scale**2))
  filter_size = filters.shape.as_list()[0]
  padding = filter_size - up_scale

  return simple_upfirdn_2d(
      image,
      filters,
      up=up_scale,
      pad0=(padding + 1) // 2 + up_scale - 1,
      pad1=padding // 2,
      device=device)


def downsample2d(image: tf.Tensor,
                 filters: tf.Tensor,
                 down_scale: int = 2,
                 gain: int = 1,
                 device: str = 'gpu') -> tf.Tensor:
  """Apply upfirdn_2d() and downsample the image."""
  filters = filters * gain
  filter_size = filters.shape.as_list()[0]
  padding = filter_size - down_scale

  return simple_upfirdn_2d(
      image,
      filters,
      down=down_scale,
      pad0=(padding + 1) // 2,
      pad1=padding // 2,
      device=device)


def upfirdn_2d(features: tf.Tensor,
               filters: tf.Tensor,
               upx: int,
               upy: int,
               downx: int,
               downy: int,
               padx0: int,
               padx1: int,
               pady0: int,
               pady1: int,
               device: str = 'gpu') -> tf.Tensor:
  """Implementation of Upsample, FIR filter, and downsample using TensorFlow ops.

  Args:
    features: A Tensor of size B x C x H x W, where B is batch size, H x W is
      the image size, and C is the number of channels.
    filters: A Tensor of size H x W x 1 x 1, H x W is the kernel size.
    upx: A integer to indicate the upsampling factor at x dimension.
    upy: A integer to indicate the upsampling factor at y dimension.
    downx: A integer to indicate the downsampling factor at x dimension.
    downy: A integer to indicate the downsampling factor at y dimension.
    padx0: A integer to indicate the padding size at left part of x dimension.
    padx1: A integer to indicate the padding size at right part of x dimension.
    pady0: A integer to indicate the padding size at left part of y dimension.
    pady1: A integer to indicate the padding size at right part of y dimension.
    device: A string specifying the type of device.

  Returns:
    outputs: The features of size B x H*Up/Down x W*Up/Down x C_out.
  """

  _, channels, height, width = features.shape.as_list()
  kernel_size = filters.shape.as_list()[0]

  # Upsample (insert zeros).
  features = tf.reshape(features, [-1, channels, height, 1, width, 1])
  features = tf.pad(
      features, [[0, 0], [0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1]])
  features = tf.reshape(features, [-1, channels, height * upy, width * upx])

  # Pad (crop if negative).
  features = tf.pad(features, [[0, 0], [0, 0], [max(
      pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)]])
  features = features[:, :,
                      max(-pady0, 0):features.shape[2] - max(-pady1, 0),
                      max(-padx0, 0):features.shape[3] - max(-padx1, 0)]

  # Convolve with filter.
  features = tf.reshape(
      features,
      [-1, 1, height * upy + pady0 + pady1, width * upx + padx0 + padx1])
  filters = filters[::-1, ::-1, None, None]

  if device == 'gpu':
    features = tf.nn.conv2d(
        features,
        filters,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NCHW')
  else:
    features = tf.transpose(features, [0, 2, 3, 1])
    features = tf.nn.conv2d(
        features,
        filters,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NHWC')
    features = tf.transpose(features, [0, 3, 1, 2])

  features = tf.reshape(features, [
      -1, channels, height * upy + pady0 + pady1 - kernel_size + 1,
      width * upx + padx0 + padx1 - kernel_size + 1
  ])

  # Downsample (throw away pixels).
  return features[:, :, ::downy, ::downx]
