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

"""Camera model used in ray generation."""

import math

import numpy as np

DEFAULT_CAMERA_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                                    [0.0, 0.0, -1.0]])
DEFAULT_CAMERA_POSITION = np.array([0.0, 0.0, 1.0])


class Camera(object):
  """A camera class that provides pixel and point projections."""

  def __init__(
      self,
      height: int = 128,
      width: int = 128,
      fov: float = 15.0,
      initial_orientation: np.ndarray = DEFAULT_CAMERA_ROTATION,
      initial_position: np.ndarray = DEFAULT_CAMERA_POSITION,
  ):
    """Constructs a camera."""
    self._image_size = (height, width)
    self._height = height
    self._width = width
    self._fov = fov
    self._initial_orientation = initial_orientation
    self._initial_position = initial_position

  def GetPixelCenters(self) -> np.ndarray:
    """Returns an [H, W, 2] array containing (x, y) half-integer pixel centers.

    Returns:
      a float32 array of shape (height, width, 2) containing (x, y) half-integer
      pixel coordinates, where (height, width) are the camera image dimensions.
    """
    shape = self._image_size
    return np.moveaxis(np.indices(shape, dtype=np.float32)[::-1], 0, -1) + 0.5

  def GetLocalRays(self) -> np.ndarray:
    """Computes the ray in the camera coordinate system.

    Note the direction vector is scaled to have unit norm.

    Returns:
      A numpy array of shape (..., 3).
    """
    x, y = np.meshgrid(
        np.linspace(-1, 1, self._width),
        np.linspace(-1, 1, self._height),
        indexing='xy')
    z = np.ones_like(x) / np.tan((2.0 * math.pi * self._fov / 360.0) / 2)
    rays_direction = np.stack([x, y, z], -1)
    rays_direction = rays_direction / np.linalg.norm(
        rays_direction, ord=2, axis=-1, keepdims=True)

    return rays_direction
