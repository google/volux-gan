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

"""Differentiable volumetric rendering implementation for pi-GAN generator."""
import math
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from generator import camera_model


_INITIAL_CAMERA_ROTATION = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_INITIAL_CAMERA_POSITION = [0.0, 0.0, 1.0]


def normalize_vecs(vectors: tf.Tensor) -> tf.Tensor:
  """Normalize vector lengths."""
  return tf.math.l2_normalize(vectors, axis=-1)


def get_initial_camera(resolution: Tuple[int, int],
                       fov: float = 12.0) -> camera_model.Camera:
  """Initialize a camera in front of the head scan and looking at it.

  Specifically, the camera is at (0,0,1) position and looking at -z axis
  direction with the subject looking at +z direction, which is consistent with
  vr/perception/deepholodeck/neural_head/data/generate_raw_image_pipeline.py.

  Args:
    resolution: Resolution of camera image in pixels.
    fov: FoV of camera.

  Returns:
    A Camera which is in front of the Holodeck 3D head scan and looking at -z
    axis, with the subject looking at +z direction.
  """
  height = resolution[0]
  width = resolution[1]

  initial_camera = camera_model.Camera(height, width, fov)
  return initial_camera


def get_rays_from_random_camera(
    batch_size: int,
    fov: float = 12.0,
    resolution: Tuple[int, int] = (64, 64),
    horizontal_stddev: float = 0.0,
    vertical_stddev: float = 0.0,
    horizontal_mean: float = math.pi * 0.5,
    vertical_mean: float = math.pi * 0.5,
    sample_distribution: str = 'normal',
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Generate rays from sampled cameras according to the given distribution.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    fov: FOV of the rendering camera.
    resolution: Resolution of camera image in pixels.
    horizontal_stddev: Standard deviation of the camera sampling in horizontal
      direction.
    vertical_stddev: Standard deviation of the camera sampling in vertical
      direction.
    horizontal_mean: Mean position of the camera sampling in horizontal
      direction.
    vertical_mean: Mean position of the camera sampling in vertical direction.
    sample_distribution: The distribution on sampling azimuth angle and
      elevation angle. Valid mode includes 'uniform', 'normal',
      'truncated_normal', 'spherical_uniform'.

  Returns:
    outputs: A list of 4 Tensors:
     - A Tensor of size b x height*width x 3, the ray directions in world
     coordinates.
     - A Tensor of size b x height*width x 3, the ray directions in camera
     coordinates.
     - A Tensor of size b x 1 x 3, the ray origins in world coordinates.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
     - A Tensor of size b x 3 x 3, the world-to-camera rotation matrix.
  """
  initial_camera = get_initial_camera(resolution, fov)

  ray_direction = initial_camera.GetLocalRays()
  ray_direction = tf.constant(ray_direction, dtype=tf.float32)

  height, width = resolution
  ray_local_direction = tf.reshape(ray_direction, (height * width, 3))

  camera_rotation, camera_position, azimuth, elevation = sample_random_camera(
      batch_size, horizontal_stddev, vertical_stddev, horizontal_mean,
      vertical_mean, sample_distribution)

  world2camera_rotation = camera_rotation
  camera2world_rotation = tf.transpose(camera_rotation, [0, 2, 1])

  ray_direction = tf.matmul(camera2world_rotation[:, None, :, :],
                            ray_local_direction[None, ..., None])[..., 0]

  ray_local_direction = ray_local_direction[None, ...]
  ray_origin = camera_position

  return (ray_direction, ray_local_direction, ray_origin, azimuth, elevation,
          world2camera_rotation)


def get_ray_points_from_sampled_camera(
    batch_size: int,
    fov: float = 12.0,
    resolution: Tuple[int, int] = (64, 64),
    horizontal_stddev: float = 0.0,
    vertical_stddev: float = 0.0,
    horizontal_mean: float = math.pi * 0.5,
    vertical_mean: float = math.pi * 0.5,
    num_samples: int = 12,
    ray_start: float = 1.85,
    ray_end: float = 2.15,
    sample_distribution: str = 'normal',
    return_rotation_matrix: bool = False,
) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                 tf.Tensor, tf.Tensor],
           Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                 tf.Tensor, tf.Tensor, tf.Tensor]]:
  """Generate points from sampled cameras according to the given distribution.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    fov: FOV of the rendering camera.
    resolution: Resolution of camera image in pixels.
    horizontal_stddev: Standard deviation of the camera sampling in horizontal
      direction.
    vertical_stddev: Standard deviation of the camera sampling in vertical
      direction.
    horizontal_mean: Mean position of the camera sampling in horizontal
      direction.
    vertical_mean: Mean position of the camera sampling in vertical direction.
    num_samples: Number of samples to take on each ray.
    ray_start: The start plane per ray,
    ray_end: The end plane per ray.
    sample_distribution: The distribution on sampling azimuth angle and
      elevation angle. Valid mode includes 'uniform', 'normal',
      'truncated_normal', 'spherical_uniform'.
    return_rotation_matrix: Indicating whether to return 3x3 rotation matrix.

  Returns:
    outputs: A list of 7 Tensors:
     - A Tensor of size b x height*width x num_samples x 3, the sampled points
     in world coordinates.
     - A Tensor of size b x height*width x num_samples x 1, the distances of the
     sampled points on each ray as a scalar.
     - A Tensor of size b x height*width x 3, the ray directions in world
     coordinates.
     - A Tensor of size b x height*width x 3, the ray directions in camera
     coordinates.
     - A Tensor of size b x 1 x 3, the ray origins in world coordinates.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
     - Optionally the world-to-camera rotation matrix of size b x 3 x 3.
  """
  (ray_direction, ray_local_direction, ray_origin, azimuth,
   elevation, camera_rotation) = get_rays_from_random_camera(
       batch_size, fov, resolution, horizontal_stddev, vertical_stddev,
       horizontal_mean, vertical_mean, sample_distribution)

  height, width = resolution
  ray_depths = tf.tile(
      tf.reshape(
          tf.linspace(ray_start, ray_end, num_samples), (1, 1, num_samples, 1)),
      (batch_size, width * height, 1, 1))

  ray_depths = perturb_points(ray_depths)

  points = ray_direction[:, :, None, :] * ray_depths
  ray_origin = ray_origin[:, None, :]
  points = ray_origin[:, :, None, :] + points

  if not return_rotation_matrix:
    return (points, ray_depths, ray_direction, ray_local_direction, ray_origin,
            azimuth, elevation)
  else:
    return (points, ray_depths, ray_direction, ray_local_direction, ray_origin,
            azimuth, elevation, camera_rotation)


def perturb_points(ray_depths: tf.Tensor) -> tf.Tensor:
  """Perturb sampling depths along each ray."""
  distance_between_points = ray_depths[:, :, 1:2, :] - ray_depths[:, :, 0:1, :]
  depth_offset = (tf.random.uniform(ray_depths.shape) -
                  0.5) * distance_between_points
  ray_depths = ray_depths + depth_offset

  return ray_depths


def sample_random_camera(
    batch_size: int,
    horizontal_stddev: float = 0.0,
    vertical_stddev: float = 0.0,
    horizontal_mean: float = math.pi * 0.5,
    vertical_mean: float = math.pi * 0.5,
    sample_distribution: str = 'normal',
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Randomly sample cameras from the given distribution.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    horizontal_stddev: Standard deviation of the camera sampling in horizontal
      direction.
    vertical_stddev: Standard deviation of the camera sampling in vertical
      direction.
    horizontal_mean: Mean position of the camera sampling in horizontal
      direction.
    vertical_mean: Mean position of the camera sampling in vertical direction.
    sample_distribution: The distribution on sampling azimuth angle and
      elevation angle. Valid mode includes 'uniform', 'normal',
      'truncated_normal', 'spherical_uniform'.

  Returns:
    outputs: A list of 4 Tensors:
     - A Tensor of size b x 3 x 3, the rotation matrices of sampled cameras.
     - A Tensor of size b x 3, the camera positions of sampled cameras.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
  """
  point_to_camera = np.array(_INITIAL_CAMERA_POSITION)
  radius = np.linalg.norm(point_to_camera)
  point_to_camera = point_to_camera / radius

  y_axis_temp = np.array([0, 1, 0])

  x_axis = np.cross(point_to_camera, y_axis_temp)
  x_axis = x_axis / np.linalg.norm(x_axis)

  y_axis = np.cross(point_to_camera, x_axis)
  rotate_matrix = np.reshape(
      np.concatenate([x_axis, y_axis, point_to_camera], axis=0), [3, 3])
  rotate_matrix = tf.constant(np.transpose(rotate_matrix), tf.float32)
  rotate_matrix = tf.tile(rotate_matrix[None, ...], (batch_size, 1, 1))
  sample_position, azimuth, elevation = random_sample_on_unit_sphere(
      batch_size=batch_size,
      horizontal_stddev=horizontal_stddev,
      vertical_stddev=vertical_stddev,
      horizontal_mean=horizontal_mean,
      vertical_mean=vertical_mean,
      sample_distribution=sample_distribution)

  sample_position = sample_position * radius
  camera_position = tf.linalg.matmul(rotate_matrix,
                                     sample_position[..., None])[..., 0]

  # Compute rotation from the location.
  z_axis = normalize_vecs(-camera_position)
  y_axis = tf.tile(
      tf.constant([0, 1, 0], dtype=tf.float32)[None, :], [batch_size, 1])
  x_axis = tf.linalg.cross(z_axis, y_axis)
  x_axis = normalize_vecs(x_axis)
  y_axis = tf.linalg.cross(z_axis, x_axis)
  camera_rotation = tf.stack([x_axis, y_axis, z_axis], axis=1)

  return camera_rotation, camera_position, azimuth, elevation


def random_sample_on_unit_sphere(
    batch_size: int,
    horizontal_stddev: float = 0.0,
    vertical_stddev: float = 0.0,
    horizontal_mean: float = math.pi * 0.5,
    vertical_mean: float = math.pi * 0.5,
    sample_distribution: str = 'normal',
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Sample points on a unit sphere according to the input distribution.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    horizontal_stddev: Standard deviation of the camera sampling in horizontal
      direction.
    vertical_stddev: Standard deviation of the camera sampling in vertical
      direction.
    horizontal_mean: Mean position of the camera sampling in horizontal
      direction.
    vertical_mean: Mean position of the camera sampling in vertical direction.
    sample_distribution: The distribution on sampling azimuth angle and
      elevation angle. Valid mode includes 'uniform', 'normal',
      'truncated_normal' and 'spherical_uniform'.

  Returns:
    outputs: A list of 3 Tensors:
     - A Tensor of size b x 3, the sampled position on the sphere.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
  """
  if sample_distribution == 'uniform':
    azimuth = (tf.random.uniform(
        (batch_size, 1)) - 0.5) * 2 * horizontal_stddev + horizontal_mean
    elevation = (tf.random.uniform(
        (batch_size, 1)) - 0.5) * 2 * vertical_stddev + vertical_mean

  elif sample_distribution == 'normal' or sample_distribution == 'gaussian':
    azimuth = tf.random.normal(
        (batch_size, 1)) * horizontal_stddev + horizontal_mean
    elevation = tf.random.normal(
        (batch_size, 1)) * vertical_stddev + vertical_mean

  elif (sample_distribution == 'truncated_gaussian' or
        sample_distribution == 'truncated_normal'):
    azimuth = tf.random.truncated_normal(
        (batch_size, 1)) * horizontal_stddev + horizontal_mean
    elevation = tf.random.truncated_normal(
        (batch_size, 1)) * vertical_stddev + vertical_mean

  elif sample_distribution == 'spherical_uniform':
    azimuth = (tf.random.uniform(
        (batch_size, 1)) - 0.5) * 2 * horizontal_stddev + horizontal_mean
    v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
    v = (tf.random.uniform((batch_size, 1)) - .5) * 2 * v_stddev + v_mean
    v = tf.clip_by_value(v, 1e-5, 1 - 1e-5)
    elevation = tf.math.acos(1 - 2 * v)
  else:
    # Just use the mean.
    azimuth = tf.ones((batch_size, 1)) * horizontal_mean
    elevation = tf.ones((batch_size, 1)) * vertical_mean

  elevation = tf.clip_by_value(elevation, 1e-3, math.pi - 1e-3)
  azimuth = tf.clip_by_value(azimuth, 1e-3, math.pi - 1e-3)

  sample_on_unit = tf.concat([
      tf.sin(elevation) * tf.cos(azimuth),
      tf.cos(elevation),
      tf.sin(elevation) * tf.sin(azimuth)
  ],
                             axis=-1)

  return sample_on_unit, azimuth, elevation


def get_rays_from_camera_with_angles(
    batch_size: int,
    azimuth: tf.Tensor,
    elevation: tf.Tensor,
    fov: float = 12.0,
    resolution: Tuple[int, int] = (64, 64),
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Generate rays of camera from given azimuth and elevation.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    azimuth: The radian to sample for azimuth.
    elevation: The radian to sample for elevation.
    fov: FOV of the rendering camera.
    resolution: Resolution of camera image in pixels.

  Returns:
    outputs: A list of 4 Tensors:
     - A Tensor of size b x height*width x 3, the ray directions in world
     coordinates.
     - A Tensor of size b x height*width x 3, the ray directions in camera
     coordinates.
     - A Tensor of size b x 1 x 3, the ray origins in world coordinates.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
     - A Tensor of size b x 3 x 3, the world-to-camera rotation matrix.
  """
  initial_camera = get_initial_camera(resolution, fov)

  ray_local_direction = initial_camera.GetLocalRays()
  ray_local_direction = tf.constant(ray_local_direction, dtype=tf.float32)

  height, width = resolution
  ray_local_direction = tf.reshape(ray_local_direction, (height * width, 3))

  camera_rotation, camera_position, azimuth, elevation = sample_camera_with_angles(
      batch_size, azimuth, elevation)

  world2camera_rotate = camera_rotation
  camera2world_rotate = tf.transpose(camera_rotation, [0, 2, 1])
  ray_direction = tf.matmul(camera2world_rotate[:, None, :, :],
                            ray_local_direction[None, ..., None])[..., 0]
  ray_local_direction = ray_local_direction[None, ...]
  ray_origin = camera_position

  return (ray_direction, ray_local_direction, ray_origin, azimuth, elevation,
          world2camera_rotate)


def sample_camera_with_angles(
    batch_size: int, azimuth: tf.Tensor,
    elevation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Sample cameras from given azimuth and elevation.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    azimuth: The radian to sample for azimuth.
    elevation: The radian to sample for elevation.

  Returns:
    outputs: A list of 4 Tensors:
     - A Tensor of size b x 3 x 3, the rotation matrices of sampled cameras.
     - A Tensor of size b x 3, the camera positions of sampled cameras.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
  """
  point_to_camera = np.array(_INITIAL_CAMERA_POSITION)
  radius = np.linalg.norm(point_to_camera)
  point_to_camera = point_to_camera / radius

  y_axis_temp = np.array([0, 1, 0])

  x_axis = np.cross(point_to_camera, y_axis_temp)
  x_axis = x_axis / np.linalg.norm(x_axis)

  y_axis = np.cross(point_to_camera, x_axis)
  rotate_matrix = np.reshape(
      np.concatenate([x_axis, y_axis, point_to_camera], axis=0), [3, 3])
  rotate_matrix = tf.constant(np.transpose(rotate_matrix), tf.float32)
  rotate_matrix = tf.tile(rotate_matrix[None, ...], (batch_size, 1, 1))

  sample_position = tf.concat([
      tf.sin(elevation) * tf.cos(azimuth),
      tf.cos(elevation),
      tf.sin(elevation) * tf.sin(azimuth)
  ],
                              axis=-1)

  sample_position = sample_position * radius
  camera_position = tf.linalg.matmul(rotate_matrix,
                                     sample_position[..., None])[..., 0]

  # Compute rotation from the location.
  z_axis = normalize_vecs(-camera_position)
  y_axis = tf.tile(
      tf.constant([0, 1, 0], dtype=tf.float32)[None, :], [batch_size, 1])
  x_axis = tf.linalg.cross(z_axis, y_axis)
  x_axis = normalize_vecs(x_axis)
  y_axis = tf.linalg.cross(z_axis, x_axis)
  camera_rotation = tf.stack([x_axis, y_axis, z_axis], axis=1)

  return camera_rotation, camera_position, azimuth, elevation


def get_ray_points_from_camera_with_angles(
    batch_size: int,
    azimuth: tf.Tensor,
    elevation: tf.Tensor,
    fov: float = 7.324,
    resolution: Tuple[int, int] = (64, 64),
    num_samples: int = 12,
    ray_start: float = 1.85,
    ray_end: float = 2.15,
    return_rotation_matrix: bool = False,
) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                 tf.Tensor, tf.Tensor],
           Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                 tf.Tensor, tf.Tensor, tf.Tensor]]:
  """Generate points from cameras sampled from given azimuth and elevation.

  Args:
    batch_size: The batch size indicating the number of camera to be sampled.
    azimuth: The radian to sample for azimuth.
    elevation: The radian to sample for elevation.
    fov: FOV of the rendering camera.
    resolution: Resolution of camera image in pixels.
    num_samples: Number of samples to take on each ray.
    ray_start: the start plane per ray,
    ray_end: the end plane per ray.
    return_rotation_matrix: Indicating whether to return 3x3 rotation matrix.

  Returns:
    outputs: A list of 7 Tensors:
     - A Tensor of size b x height*width x num_samples x 3, the sampled points
     in world coordinates.
     - A Tensor of size b x height*width x num_samples x 1, the distances of the
     sampled points on each ray as a scalar.
     - A Tensor of size b x height*width x 3, the ray directions in world
     coordinates.
     - A Tensor of size b x height*width x 3, the ray directions in camera
     coordinates.
     - A Tensor of size b x 1 x 3, the ray origins in world coordinates.
     - A Tensor of size b x 1, the azimuth angle of sampled cameras.
     - A Tensor of size b x 1, the elevation angle of sampled cameras.
     - Optionally the world-to-camera rotation matrix of size b x 3 x 3.

  """
  (ray_direction, ray_local_direction, ray_origin, azimuth, elevation,
   camera_rotation) = get_rays_from_camera_with_angles(batch_size, azimuth,
                                                       elevation, fov,
                                                       resolution)

  height, width = resolution
  ray_depths = tf.tile(
      tf.reshape(
          tf.linspace(ray_start, ray_end, num_samples), (1, 1, num_samples, 1)),
      (batch_size, width * height, 1, 1))

  ray_depths = perturb_points(ray_depths)

  points = ray_direction[:, :, None, :] * ray_depths
  ray_origin = ray_origin[:, None, :]
  points = ray_origin[:, :, None, :] + points

  if not return_rotation_matrix:
    return (points, ray_depths, ray_direction, ray_local_direction, ray_origin,
            azimuth, elevation)
  else:
    return (points, ray_depths, ray_direction, ray_local_direction, ray_origin,
            azimuth, elevation, camera_rotation)
