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

"""Models for embeddings e.g., a positional embedding."""
from typing import Optional

import tensorflow as tf


class SinusoidalEncoder(tf.keras.layers.Layer):
  """Sinusoidal Positional Encodings."""

  def __init__(self,
               num_freqs: int,
               max_freq_log2: Optional[float] = None,
               scale: float = 1.0):
    """Initializes the sinusoidal positional Encoder."""
    super().__init__()

    if max_freq_log2 is None:
      max_freq_log2 = num_freqs - 1.0

    self._num_freqs = num_freqs
    self._freq_bands = 2.0**tf.linspace(0.0, max_freq_log2, num_freqs)
    self._scale = scale

  @property
  def output_dim(self) -> int:
    """Return the dimension of the encoded feature."""
    return 2 * self._num_freqs + 1

  def call(self, features: tf.Tensor) -> tf.Tensor:
    """A vectorized sinusoidal encoding.

    Args:
      features: the input features to encode.

    Returns:
      A tensor containing the encoded features.
    """
    freqs = self._freq_bands
    batch_shape = features.shape[:-1]
    batch_ones = [1] * len(batch_shape)

    freqs = tf.reshape(freqs, (*batch_ones, self._num_freqs, 1))  # (*, F, 1).
    feature_expanded = tf.expand_dims(features, axis=-2)  # (*, 1, C).
    # Will be broadcasted to shape (*B, F, C).
    angles = self._scale * feature_expanded * freqs

    # The shape of the features is (*B, F, 2, C) so that when we reshape it
    # it matches the ordering of the original NeRF code.
    sin_features = tf.stack((tf.sin(angles), tf.cos(angles)), axis=-2)
    sin_features = tf.reshape(sin_features, (*batch_shape, -1))

    # Prepend the original signal for the identity.
    features = tf.concat([features, sin_features], axis=-1)
    return features
