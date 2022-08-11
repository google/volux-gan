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

"""Differentiable volumetric implementation used by pi-GAN generator."""
from typing import Tuple

import tensorflow as tf

_EPSILON = 1e-6


def render_radiance_field(
    point_colors: tf.Tensor,
    point_densities: tf.Tensor,
    point_distances: tf.Tensor,
    noise_std: float = 0.5,
    clamp_mode: str = 'softplus',
    sample_at_infinity: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Renders a 3D radiance field along rays.

  This function implements volume rendering of 3D radiance field (with per point
  color and density) from samples of the radiance field along rays.

  Args:
    point_colors: Shape [..., num_samples_per_ray, 3]. RGB color values at all
      point locations along the rays. These should be in [0, 1] (typically they
      would be post-sigmoid activations).
    point_densities: Shape [..., num_samples_per_ray, 1]. Sigma values at all
      point locations along the rays.  These have arbitrary range ([-inf,inf])
      and can be straight from a network.
    point_distances: Shape [..., num_samples_per_ray, 1]. Distances of each
      point along the rays.
    noise_std: Gaussian noise to perturb densities.
    clamp_mode: Functions used to clamp densities, valid mode includes
      'softplus', 'relu'.
    sample_at_infinity: If True samples the last point at infinity.

  Returns:
    Tensor with shape [..., 3]. Estimated RGB color of a ray.
    Tensor with shape [..., 1]. Distance map. Distance expectation of a ray.
    Tensor with shape [..., num_samples, 1]. Weights assigned to each 3D points.
  """
  # Compute 'distance' (in time) between each integration time along a ray.
  delta_distance = point_distances[..., 1:, :] - point_distances[..., :-1, :]

  # The 'distance' from the last integration time is infinity.
  last_sample_dist = 1e10 if sample_at_infinity else 1e-12
  delta_distance = tf.concat([
      delta_distance,
      tf.broadcast_to([last_sample_dist], delta_distance[..., :1, :].shape)
  ],
                             axis=-2)  # [N_rays, num_samples]
  delta_distance = delta_distance + 1e-8

  noise = tf.random.normal(point_densities.shape) * noise_std

  if clamp_mode == 'softplus':
    alphas = 1.0 - tf.exp(
        -delta_distance * tf.math.softplus(point_densities + noise))
  elif clamp_mode == 'relu':
    alphas = 1.0 - tf.exp(-delta_distance * tf.nn.relu(point_densities + noise))
  else:
    raise ValueError('Need to choose clamp mode')

  transmittances = tf.exp(-tf.cumsum(alphas, axis=-2, exclusive=True))
  weights = alphas * transmittances

  rgb_final = tf.reduce_sum(weights * point_colors, axis=-2)
  depth_final = tf.reduce_sum(weights * point_distances, axis=-2)

  return rgb_final, depth_final, weights


def sample_pdf(
    bins: tf.Tensor,
    weights: tf.Tensor,
    num_samples: int,
    uniform_sampling=False,
) -> tf.Tensor:
  """Sample fine samples from bins with distribution.

  Args:
    bins: Positions of the bins.
    weights: Weights per bin.
    num_samples: The number of samples.
    uniform_sampling: Whether to uniformly sample the bins.

  Returns:
    Tensor with shape [..., 1], the sampled distance values.
  """
  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0.
  weights += _EPSILON  # Prevent nans
  pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
  cdf = tf.cumsum(pdf, -1)
  cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

  # Take uniform samples.
  if uniform_sampling:
    uniform_samples = tf.linspace(0., 1., num_samples)
    uniform_samples = tf.broadcast_to(uniform_samples,
                                      list(cdf.shape[:-1]) + [num_samples])
  else:
    uniform_samples = tf.random.uniform(list(cdf.shape[:-1]) + [num_samples])

  # Invert CDF.
  bin_indices = tf.searchsorted(cdf, uniform_samples, side='right')
  below_bin_id = tf.maximum(0, bin_indices - 1)
  above_bin_id = tf.minimum(cdf.shape[-1] - 1, bin_indices)
  below_above_bin_id = tf.stack([below_bin_id, above_bin_id], -1)
  below_above_bin_cdf = tf.gather(
      cdf,
      below_above_bin_id,
      axis=-1,
      batch_dims=len(below_above_bin_id.shape) - 2)
  below_above_bins = tf.gather(
      bins,
      below_above_bin_id,
      axis=-1,
      batch_dims=len(below_above_bin_id.shape) - 2)

  denom = (below_above_bin_cdf[..., 1] - below_above_bin_cdf[..., 0])
  denom = tf.where(denom < _EPSILON, tf.ones_like(denom), denom)
  bin_prob = (uniform_samples - below_above_bin_cdf[..., 0]) / denom
  samples = below_above_bins[..., 0] + bin_prob * (
      below_above_bins[..., 1] - below_above_bins[..., 0])

  return samples
