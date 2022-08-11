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

"""Relit Library for VoLux-GAN."""

import math
from typing import Callable, Tuple, Dict, Sequence

import tensorflow as tf
import tensorflow_addons.image as tfa_image

_SPECULAR_CLIP_MAX = 10.0


def _safe_acos(x: tf.Tensor, min_grad: float = -5.0) -> tf.Tensor:
  """A numerically safe version of the acos function.

  While acos(x) is in [0,pi] the gradient of acos(x) goes to -infintiy for
  argument values 1.0 and -1.0. This version of the acos function clips the
  gradient at a minimal value.

  Args:
    x: The input tensor to the safe acos function.
    min_grad: The minimal gradient value allowed.

  Returns:
    The forward pass output of the safe_acos function.
  """

  # The inner function is used to close over min_grad. Directly adding
  # the min_grad argument here is only supported in eager mode and fails if
  # used together with tf.function().
  @tf.custom_gradient
  def safe_acos_implementation(
      x: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
    """The implementation of the numerically safe version of the acos function.

    Args:
      x: The input tensor to the safe acos function.

    Returns:
      The forward pass output of the acos function and a callable that computes
      the gradient.
    """
    y = tf.math.acos(x)

    def grad(dy):
      return dy * tf.maximum(min_grad, -1 / tf.sqrt(1 - tf.square(x)))

    return y, grad

  return safe_acos_implementation(x)


def _get_light_visibility(directions: tf.Tensor,
                          hdr_map: tf.Tensor) -> tf.Tensor:
  """Computes a light visibility using direction vectors and a HDR map."""
  directions = tf.clip_by_value(directions, -1.0, 1.0)

  x = directions[..., 0]
  y = directions[..., 1]
  z = directions[..., 2]

  phis = -tf.math.atan2(x, z)
  thetas = _safe_acos(y, min_grad=-5.0)

  phis += math.pi

  pixels_height, pixels_width = tf.unstack(tf.shape(hdr_map))[1:3]
  pixels_height = tf.cast(pixels_height, dtype=tf.float32)
  pixels_width = tf.cast(pixels_width, dtype=tf.float32)

  pixels_x = ((phis * pixels_width) / (2 * math.pi))
  pixels_y = ((thetas * pixels_height) / math.pi)

  pixels_x = tf.clip_by_value(pixels_x, 0, pixels_width - 1)
  pixels_y = tf.clip_by_value(pixels_y, 0, pixels_height - 1)

  coordinates = tf.stack((pixels_x, pixels_y), axis=-1)

  light_visibility = tfa_image.resampler(hdr_map, coordinates)

  return light_visibility


def compute_specular_light_visibility(normals: tf.Tensor, hdr_map: tf.Tensor,
                                      camera_orientations: tf.Tensor,
                                      clip: bool) -> tf.Tensor:
  """Computes specular RGB light visibility from normals and a HDR map.

  Args:
    normals: Normal map expected to be in the range [-1, 1].
    hdr_map: The HDR environment in latlong format. Note that the HDR map is
      expected to be preconvolved to apply the Phong reflection model with a
      particaular specular exponent.
    camera_orientations: The camera orientations needed to rotate the normals
      back to world space, to which hdr_map is aligned.
    clip: Whether to clip the final light map to a specified max value.

  Returns:
    A RGB specular light visibility image.
  """
  normals = tf.clip_by_value(normals, -1., 1.)

  # Negates the normals directions axes to correspond to camera space.
  normals = tf.concat([
      tf.expand_dims(normals[:, :, :, 0], -1),
      -tf.expand_dims(normals[:, :, :, 1], -1),
      -tf.expand_dims(normals[:, :, :, 2], -1),
  ],
                      axis=-1)

  # Compute the reflected directions.
  camera_directions = tf.reshape(
      tf.constant([0, 0, 1], dtype=tf.float32), (1, 1, 1, 3))
  batch, height, width = tf.unstack(tf.shape(normals))[:3]
  camera_directions = tf.tile(camera_directions, (batch, height, width, 1))
  reflected_directions = camera_directions - (
      2 * normals *
      (tf.reduce_sum(camera_directions * normals, axis=-1, keepdims=True)))

  # Rotates directions with the camera rotation matrix to be in world space.
  reflected_directions = tf.reshape(reflected_directions,
                                    (batch, height * width, 3))
  reflected_directions = tf.linalg.matmul(reflected_directions,
                                          camera_orientations)
  reflected_directions = tf.reshape(reflected_directions,
                                    (batch, height, width, 3))

  light_visibility = _get_light_visibility(reflected_directions, hdr_map)

  if clip:
    clip_max = _SPECULAR_CLIP_MAX
    light_visibility = tf.clip_by_value(light_visibility, 0.0, clip_max)

  return light_visibility


def compute_diffuse_light_visibility(
    normals: tf.Tensor, hdr_map: tf.Tensor,
    camera_orientations: tf.Tensor) -> tf.Tensor:
  """Computes diffuse RGB light visibility from normals and a HDR map.

  Args:
    normals: Normal map expected to be in the range [-1, 1].
    hdr_map: The HDR environment in latlong format. Note that the HDR map here
      is expected to have already been diffuse convolved to approximate the
      integration across multiple directions.
    camera_orientations: The camera orientations needed to rotate the normals
      back to world space, to which hdr_map is aligned.

  Returns:
    A RGB diffuse light visibility image.
  """
  normals = tf.clip_by_value(normals, -1., 1.)

  # Negates the normals directions axes to correspond to camera space.
  normals = tf.concat([
      tf.expand_dims(normals[:, :, :, 0], -1),
      -tf.expand_dims(normals[:, :, :, 1], -1),
      -tf.expand_dims(normals[:, :, :, 2], -1),
  ],
                      axis=-1)

  # Rotates directions with the camera rotation matrix to be in world space.
  batch, height, width = tf.unstack(tf.shape(normals))[:3]
  normals = tf.reshape(normals, (batch, height * width, 3))
  normals = tf.linalg.matmul(normals, camera_orientations)
  normals = tf.reshape(normals, (batch, height, width, 3))

  return _get_light_visibility(normals, hdr_map)


def postpreocess_specular_visibility(
    specular_light_visibility: tf.Tensor) -> tf.Tensor:
  """Post-process the value of specular visibility in log scales."""
  specular_light_visibility = tf.clip_by_value(specular_light_visibility, 0.0,
                                               _SPECULAR_CLIP_MAX)
  specular_light_visibility = tf.math.log1p(specular_light_visibility) / 2.0
  return specular_light_visibility


def combine_specular_light_visibilities(
    specular_light_visibility_1: tf.Tensor,
    specular_light_visibility_16: tf.Tensor,
    specular_light_visibility_32: tf.Tensor,
    specular_light_visibility_64: tf.Tensor,
    reflection_ratios: tf.Tensor) -> tf.Tensor:
  """Combines the specular light visibilities."""
  # Normalizes the ratios to sum to 1.
  reflection_ratios_list = tf.split(reflection_ratios, 4, axis=-1)

  specular_light_visibility = (
      specular_light_visibility_1 * reflection_ratios_list[0] +
      specular_light_visibility_16 * reflection_ratios_list[1] +
      specular_light_visibility_32 * reflection_ratios_list[2] +
      specular_light_visibility_64 * reflection_ratios_list[3])

  specular_light_visibility = postpreocess_specular_visibility(
      specular_light_visibility)

  return specular_light_visibility


def get_specular_light_visibility(
    predicted_normals: tf.Tensor, diffuse_hdr_map: tf.Tensor,
    specular_hdr_map_16: tf.Tensor, specular_hdr_map_32: tf.Tensor,
    specular_hdr_map_64: tf.Tensor,
    camera_orientation: tf.Tensor) -> Sequence[tf.Tensor]:
  """Gets the specular light visibility corresponding to the given params."""
  specular_light_visibility_1 = compute_specular_light_visibility(
      predicted_normals, diffuse_hdr_map, camera_orientation, False)
  specular_light_visibility_16 = compute_specular_light_visibility(
      predicted_normals, specular_hdr_map_16, camera_orientation, False)
  specular_light_visibility_32 = compute_specular_light_visibility(
      predicted_normals, specular_hdr_map_32, camera_orientation, False)
  specular_light_visibility_64 = compute_specular_light_visibility(
      predicted_normals, specular_hdr_map_64, camera_orientation, False)

  outputs = [
      specular_light_visibility_1, specular_light_visibility_16,
      specular_light_visibility_32, specular_light_visibility_64
  ]

  return outputs


def run_relit_preprocess(
    normal_maps: tf.Tensor, hdr_maps: Dict[str, tf.Tensor],
    camera_orientations: tf.Tensor) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
  """Computes diffuse and specular light visibility from normals and a HDR map.

  Args:
    normal_maps: Normal map expected to be in the range [-1, 1].
    hdr_maps: A dict of HDR environments in latlong format. Note that the HDR
      maps are expected to be preconvolved to apply the Phong reflection model
      with a particaular specular exponent.
    camera_orientations: The camera orientations needed to rotate the normals
      back to world space, to which hdr_map is aligned.

  Returns:
    A tuple of two containing the following two tensors.
    - A RGB diffuse light visibility image.
    - A list containing 4 RGB specular light visibility images in different
    specular exponents.
  """
  diffuse_hdr_map = hdr_maps['diffuse_hdr_map']
  specular_hdr_map_16 = hdr_maps['specular_hdr_map_16']
  specular_hdr_map_32 = hdr_maps['specular_hdr_map_32']
  specular_hdr_map_64 = hdr_maps['specular_hdr_map_64']

  diffuse_light_visibility = compute_diffuse_light_visibility(
      normal_maps, diffuse_hdr_map, camera_orientations)

  specular_visib_outputs = get_specular_light_visibility(
      normal_maps, diffuse_hdr_map, specular_hdr_map_16, specular_hdr_map_32,
      specular_hdr_map_64, camera_orientations)

  return (diffuse_light_visibility, specular_visib_outputs)
