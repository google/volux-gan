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

"""Implicit generator for Relightable 3D Face Generation."""
import math
from typing import Optional, Tuple, Union, Dict, Sequence

import tensorflow as tf
import tensorflow_addons.image as tfa_image
from generator import ray_generator
from generator import relit_lib
from generator import volumetric_rendering
from generator.network import nerf_lib
from generator.network import resunet_extractor_lib

_MAPPING_LAYER_DIMENSION = 512
_NERF_LAYER_DIMENSION = 256


class ImplicitGenerator3d(tf.keras.Model):
  """Build the Relightable 3D Face Generation Model."""

  def __init__(
      self,
      latent_dim: int,
      upsample_factor: int = 4,
      device: str = 'gpu',
  ):
    """Initialize the 3D Face Generator.

    Args:
      latent_dim: An integer specifying the dimension of the random sampled
        latent code.
      upsample_factor: An integer specifying the upsampling factor.
      device: A string specifying the type of device.
    """
    super().__init__()
    self._return_normal = (device == 'gpu')
    self.nerf_mlps = nerf_lib.NeRFBaseline(
        hidden_dim=_NERF_LAYER_DIMENSION, device=device)
    self.conv_network = nerf_lib.StyleConvNetwork(
        start_dim=_NERF_LAYER_DIMENSION, upsample_factor=4, device=device)
    self.mapping_network = nerf_lib.CustomMappingNetwork(
        hidden_dim=_MAPPING_LAYER_DIMENSION,
        nerf_dim=_NERF_LAYER_DIMENSION,
        conv_start_dim=_NERF_LAYER_DIMENSION,
        upsample_factor=4)

    self.latent_dim = latent_dim
    self.upsample_factor = upsample_factor
    self.style_mixing_prob = 0.9
    self.mixing_layer_indices = tf.range(
        self.mapping_network.total_broadcast, dtype=tf.int32)[tf.newaxis, :,
                                                              tf.newaxis]

    self.render_bottle_neck_mlps_3 = resunet_extractor_lib.ResUNet(
        [64, 128, 256], output_channel=6)

  def call(
      self,
      latent_code: tf.Tensor,
      nerf_resolution: Tuple[int, int] = (64, 64),
      fov: float = 13.0,
      ray_start: float = 0.0,
      ray_end: float = 2.0,
      num_samples: int = 16,
      hierarchical_num_samples: int = 12,
      h_stddev: float = 1.0,
      v_stddev: float = 1.0,
      h_mean: float = math.pi * 0.5,
      v_mean: float = math.pi * 0.5,
      sample_dist: Optional[str] = None,
      has_background: bool = False,
      white_back: bool = False,
      sample_at_infinity: bool = False,
      clamp_mode: str = 'relu',
      nerf_noise: float = 0.5,
      return_mapping_embeddings: bool = False,
      **kwargs
  ) -> Union[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[
      str, tf.Tensor], Dict[str, tf.Tensor], tf.Tensor], Tuple[Dict[
          str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[
              str, tf.Tensor], tf.Tensor, tf.Tensor]]:
    """Run a forward pass to sample camera and render the intrinsic images.

    Args:
      latent_code: A Tensor of size B x C, where B is the batch size, C is the
        dimension of the random sampled latent code.
      nerf_resolution: A typle of integer specifying the output resolution of
        NeRF.
      fov: A float specifying the field of view of the virtual camera.
      ray_start: A float specifying the start depth of the sampling range.
      ray_end: A float specifying the end depth of the sampling range.
      num_samples: An integer specifying the number of rough samples.
      hierarchical_num_samples: An integer specifying the number of hierarchical
        samples.
      h_stddev: A float specifying the vertical deviation of random sampled
        camera.
      v_stddev: A float specifying the horizontal deviation of random sampled
        camera.
      h_mean: A float specifying the vertical mean of random sampled camera.
      v_mean: A float specifying the horizontal mean of random sampled camera.
      sample_dist: A string specifying the sampling distribution of the camera.
      has_background: A bool indicating whether the rendering has background.
      white_back: A bool indicating whether the background is black or white.
      sample_at_infinity: A bool indicating whether sampling the infinite point.
      clamp_mode: A string specifying the clamp function of the alpha.
      nerf_noise: A float specifying the noise added to the alpha.
      return_mapping_embeddings: A bool indicating whether to return mapping
        embeddings.
      **kwargs: Keyworded arguments that are forwarded by the function.

    Returns:
      A tuple of Tensors:
        - Albedo images in different resolutions.
        - Depth images.
        - Normal maps.
        - Foreground maks.
        - Angles of the sampled camera.
        Optional:
        - Mapping embeddings.
    """
    batch_size = latent_code.shape.as_list()[0]

    mapping_embeddings = self.mapping_network.inference_mapping_embeddings(
        latent_code)
    mapping_embeddings = self.mapping_network.broadcast(mapping_embeddings)
    mapping_embeddings = self.mix_mapping_embeddings(latent_code,
                                                     mapping_embeddings)

    (nerf_style_feature, conv_style_feature
    ) = self.mapping_network.inference_style_vectors(mapping_embeddings)

    # Generate initial camera rays and sample points.
    (points, z_vals, ray_directions, ray_local_directions, ray_origins, azimuth,
     elevation,
     world2camera_rotation) = ray_generator.get_ray_points_from_sampled_camera(
         batch_size,
         fov,
         nerf_resolution,
         h_stddev,
         v_stddev,
         h_mean,
         v_mean,
         num_samples,
         ray_start,
         ray_end,
         sample_dist,
         return_rotation_matrix=True)

    rgb, depths, normals, masks, angles = self.inference_from_style_embeddings(
        nerf_style_feature=nerf_style_feature,
        conv_style_feature=conv_style_feature,
        points=points,
        z_vals=z_vals,
        ray_directions=ray_directions,
        ray_local_directions=ray_local_directions,
        ray_origins=ray_origins,
        azimuth=azimuth,
        elevation=elevation,
        world2camera_rotation=world2camera_rotation,
        batch_size=batch_size,
        img_size=nerf_resolution,
        ray_start=ray_start,
        ray_end=ray_end,
        num_samples=num_samples,
        hierarchical_num_samples=hierarchical_num_samples,
        white_back=white_back,
        fov=fov,
        has_background=has_background,
        sample_at_infinity=sample_at_infinity,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
    )
    if not return_mapping_embeddings:
      return (rgb, depths, normals, masks, angles)
    else:
      return (rgb, depths, normals, masks, angles, mapping_embeddings)

  def inference_from_style_embeddings(
      self,
      nerf_style_feature: Sequence[tf.Tensor],
      conv_style_feature: Sequence[tf.Tensor],
      points: tf.Tensor,
      z_vals: tf.Tensor,
      ray_directions: tf.Tensor,
      ray_local_directions: tf.Tensor,
      ray_origins: tf.Tensor,
      azimuth: tf.Tensor,
      elevation: tf.Tensor,
      world2camera_rotation: tf.Tensor,
      batch_size: int,
      img_size: Tuple[int, int] = (128, 128),
      ray_start: float = 0.0,
      ray_end: float = 2.0,
      num_samples: int = 16,
      hierarchical_num_samples: int = 12,
      white_back: bool = False,
      fov: float = 13.0,
      has_background: bool = False,
      sample_at_infinity: bool = False,
      clamp_mode: str = 'relu',
      nerf_noise: float = 0.5,
      **kwargs
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor],
             Dict[str, tf.Tensor], tf.Tensor]:
    """Run a forward pass of NeRF and Upsampling module.

    Args:
      nerf_style_feature: A list tensor of size B x C, where B is the batch
        size, C is the dimension of the mapping embedding of NeRF.
      conv_style_feature: A list tensor of size B x C, where B is the batch
        size, C is the dimension of the mapping embedding of Upsampler.
      points: A tensor of size B x H x W x N x 3, where B is the batch size, H
        is the height, W is the width, N is the sampling number.
      z_vals: A tensor of size B x H x W x N, where B is the batch size, H is
        the height, W is the width, N is the sampling number.
      ray_directions: A tensor of size B x H x W x 3, where B is the batch size,
        H is the height, W is the width.
      ray_local_directions: A tensor of size B x H x W x 3, where B is the batch
        size, H is the height, W is the width.
      ray_origins: A tensor of size B x H x W x 3, where B is the batch size, H
        is the height, W is the width.
      azimuth: A Tensor of size B x 1 specifying the azimuth angle of random
        sampled camera.
      elevation: A Tensor of size B x 1 specifying the elevation angle of random
        sampled camera.
      world2camera_rotation: A Tensor of size B x 3 x 3 specifying the rotation
        matrix of random sampled camera.
      batch_size: An integer specifying the batch size.
      img_size: A tuple of integers specifying the resolution of rendering.
      ray_start: A float specifying the start depth of the sampling range.
      ray_end: A float specifying the end depth of the sampling range.
      num_samples: An integer specifying the number of rough samples.
      hierarchical_num_samples: An integer specifying the number of hierarchical
        samples.
      white_back: bool = False,
      fov: A float specifying the field of view of the virtual camera.
      has_background: A bool indicating whether the rendering has background.
      sample_at_infinity: A bool indicating whether sampling the infinite point.
      clamp_mode: A string specifying the clamp function of the alpha.
      nerf_noise: A float specifying the noise added to the alpha.
      **kwargs: Keyworded arguments that are forwarded by the function.

    Returns:
      A tuple of Tensors:
        - Albedo images in different resolutions.
        - Depth images.
        - Normal maps.
        - Foreground maks.
        - Angles of the sampled camera.
    """
    self.nerf_resolution = img_size
    self.high_resolution = [
        img_size[0] * self.upsample_factor, img_size[1] * self.upsample_factor
    ]
    ray_directions_expanded = tf.tile(
        tf.expand_dims(ray_directions, axis=-2), (1, 1, num_samples, 1))
    ray_directions_expanded = tf.reshape(
        ray_directions_expanded,
        (batch_size, img_size[0] * img_size[1] * num_samples, 3))

    bg_points = tf.expand_dims(
        ray_origins, axis=-2) + tf.expand_dims(
            ray_directions, axis=-2) * tf.ones([1, 1, 1, 3]) * 1.5
    points = tf.concat([points, bg_points], axis=-2)
    points = tf.reshape(points, (batch_size, img_size[0] * img_size[1] *
                                 (num_samples + 1), 3))

    coarse_output, coarse_normal = self.nerf_mlps(
        [points, nerf_style_feature], return_normal=self._return_normal)

    coarse_output = tf.reshape(
        coarse_output,
        (batch_size, img_size[0] * img_size[1], num_samples + 1, 264))
    coarse_normal = tf.reshape(
        coarse_normal,
        (batch_size, img_size[0] * img_size[1], num_samples + 1, 3))

    coarse_output = coarse_output[:, :, :-1]
    coarse_normal = coarse_normal[:, :, :-1]
    bg_rgb_intrinsic_feature = coarse_output[:, :, -1:, :263]

    coarse_rgb_intrinsic_feature = coarse_output[..., :263]
    coarse_density = coarse_output[..., 263:]

    self.camera_orientations = world2camera_rotation
    initial_camera = tf.constant([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                                  [0.0, 0.0, -1.0]])
    world2camera_rotation = -tf.matmul(initial_camera, world2camera_rotation)
    coarse_normal = tf.matmul(world2camera_rotation[:, None, None, :, :],
                              coarse_normal[:, :, :, :, None])[..., 0]
    coarse_rgb_intrinsic_feature_normal = tf.concat(
        [coarse_rgb_intrinsic_feature, coarse_normal], axis=-1)

    if hierarchical_num_samples > 0:
      _, _, weights = volumetric_rendering.render_radiance_field(
          coarse_rgb_intrinsic_feature_normal,
          coarse_density,
          z_vals,
          noise_std=nerf_noise,
          clamp_mode=clamp_mode,
          sample_at_infinity=sample_at_infinity)

      weights = tf.reshape(
          weights, (batch_size * img_size[0] * img_size[1], num_samples))

      #### Start new importance sampling
      z_vals = tf.reshape(z_vals,
                          (batch_size * img_size[0] * img_size[1], num_samples))
      z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
      z_vals = tf.reshape(
          z_vals, (batch_size, img_size[0] * img_size[1], num_samples, 1))
      fine_z_vals = volumetric_rendering.sample_pdf(
          z_vals_mid,
          weights[:, 1:-1] + 1e-8,
          hierarchical_num_samples,
          uniform_sampling=False)
      fine_z_vals = tf.reshape(
          fine_z_vals,
          (batch_size, img_size[0] * img_size[1], hierarchical_num_samples, 1))

      fine_points = tf.expand_dims(
          ray_origins, axis=-2) + tf.expand_dims(
              ray_directions, axis=-2) * tf.tile(fine_z_vals, (1, 1, 1, 3))
      fine_points = tf.reshape(
          fine_points,
          (batch_size, img_size[0] * img_size[1] * hierarchical_num_samples, 3))

      #### end new importance sampling
      fine_points = tf.stop_gradient(fine_points)
      fine_output, fine_normal = self.nerf_mlps(
          [fine_points, nerf_style_feature], return_normal=self._return_normal)

      fine_output = tf.reshape(fine_output,
                               (batch_size, img_size[0] * img_size[1], -1, 264))
      fine_normal = tf.reshape(fine_normal,
                               (batch_size, img_size[0] * img_size[1], -1, 3))

      fine_rgb_intrinsic_feature = fine_output[..., :263]
      fine_density = fine_output[..., 263:]
      fine_normal = tf.matmul(world2camera_rotation[:, None, None, :, :],
                              fine_normal[:, :, :, :, None])[..., 0]
      fine_rgb_intrinsic_feature_normal = tf.concat(
          [fine_rgb_intrinsic_feature, fine_normal], axis=-1)

      # Combine course and fine points
      all_z_vals = tf.concat([fine_z_vals, z_vals], -2)
      all_rgb_feature_intrinsic_normal = tf.concat([
          fine_rgb_intrinsic_feature_normal, coarse_rgb_intrinsic_feature_normal
      ], -2)
      all_density = tf.concat([fine_density, coarse_density], -2)

      sort_indice = tf.argsort(
          all_z_vals, axis=-2, direction='ASCENDING')[..., 0]
      all_z_vals = tf.gather(all_z_vals, sort_indice, axis=-2, batch_dims=2)
      all_rgb_feature_intrinsic_normal = tf.gather(
          all_rgb_feature_intrinsic_normal, sort_indice, axis=-2, batch_dims=2)
      all_density = tf.gather(all_density, sort_indice, axis=-2, batch_dims=2)

      sort_indice = tf.argsort(
          fine_z_vals, axis=-2, direction='ASCENDING')[..., 0]
      fine_z_vals = tf.gather(fine_z_vals, sort_indice, axis=-2, batch_dims=2)
      fine_rgb_intrinsic_feature_normal = tf.gather(
          fine_rgb_intrinsic_feature_normal, sort_indice, axis=-2, batch_dims=2)
      fine_density = tf.gather(fine_density, sort_indice, axis=-2, batch_dims=2)
    else:
      all_z_vals = z_vals
      all_rgb_feature_intrinsic_normal = coarse_rgb_intrinsic_feature_normal
      all_density = coarse_density

    all_normals = all_rgb_feature_intrinsic_normal[..., 263:]
    all_normals = tf.reshape(
        all_normals,
        [batch_size, self.nerf_resolution[0], self.nerf_resolution[0], -1])
    all_normals = tfa_image.gaussian_filter2d(
        all_normals, filter_shape=(3, 3), sigma=1)
    all_normals = tf.reshape(
        all_normals,
        [batch_size, self.nerf_resolution[0] * self.nerf_resolution[0], -1, 3])
    all_rgb_feature_intrinsic_normal = tf.concat(
        [all_rgb_feature_intrinsic_normal[..., :-3], all_normals], axis=-1)

    self._all_normals = all_normals
    self._all_intrinsic = all_rgb_feature_intrinsic_normal[..., 3:7]
    self._all_density = all_density
    self._all_z_vals = all_z_vals
    self._sample_at_infinity = sample_at_infinity
    self._clamp_mode = clamp_mode

    # Create images with NeRF
    (rgb_intrinsic_feature_normal, depths,
     weights) = volumetric_rendering.render_radiance_field(
         all_rgb_feature_intrinsic_normal,
         all_density,
         all_z_vals,
         sample_at_infinity=sample_at_infinity,
         clamp_mode=clamp_mode,
         noise_std=nerf_noise)
    rgb_intrinsic_feature_normal = tf.reshape(
        rgb_intrinsic_feature_normal,
        (batch_size, img_size[0], img_size[1], 266))

    bg_rgb_intrinsic_feature = tf.reshape(
        bg_rgb_intrinsic_feature, (batch_size, img_size[0], img_size[1], 263))

    weights = tf.reshape(weights, (batch_size, img_size[0], img_size[1], -1))
    weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)

    low_rgb = rgb_intrinsic_feature_normal[..., :3]

    if white_back:
      low_rgb = 1 - low_rgb
    low_rgb = low_rgb * 2.0 - 1.0

    feature = rgb_intrinsic_feature_normal[..., 7:263]
    feature = feature + (1 - weights_sum) * bg_rgb_intrinsic_feature[:, :, :,
                                                                     7:]

    regressed_normal = rgb_intrinsic_feature_normal[..., 263:]
    regressed_normal = tf.math.l2_normalize(regressed_normal, axis=-1)

    ray_local_directions = tf.reshape(ray_local_directions,
                                      (1, img_size[0], img_size[1], 3))
    depths = tf.reshape(depths, (batch_size, img_size[0], img_size[1], 1))
    depths = depths * ray_local_directions[..., 2:]

    high_rgb, conv_feature = self.conv_network([feature, conv_style_feature])

    weight_mask = tf.image.resize(
        weights_sum, (self.high_resolution[0], self.high_resolution[1]),
        'bilinear')
    high_rgb[-1] = (high_rgb[-1] + 1) * tf.cast(weight_mask > 0.1,
                                                tf.float32) - 1

    self.accumulated_feature = conv_feature

    weights, _ = tf.math.top_k(weights, k=6)
    weights_max_sum = tf.math.reduce_sum(weights, axis=-1, keepdims=True)

    rgb = {
        'low': low_rgb,
        'high': high_rgb,
    }
    depths = {
        'nerf': depths,
        'conv': depths,
    }
    normals = {
        'full': regressed_normal,
    }
    masks = {
        'nerf': weights_sum,
        'conv': weights_sum,
        'max_sum': weights_max_sum,
    }

    return rgb, depths, normals, masks, tf.concat([azimuth, elevation], axis=-1)

  def infer_relit(
      self,
      rgb: Dict[str, tf.Tensor],
      normals: Dict[str, tf.Tensor],
      masks: Dict[str, tf.Tensor],
      hdri_map: Dict[str, tf.Tensor],
      camera_orientations: Optional[tf.Tensor] = None
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Predict relit image from input albedo, normals and HDR components."""
    high_image = tf.stop_gradient(rgb['high'][-1])
    input_mask = tf.stop_gradient(masks['nerf'])
    batch_size = high_image.shape.as_list()[0]

    input_mask = tf.image.resize(
        input_mask, (self.high_resolution[0], self.high_resolution[1]),
        'bilinear')

    if camera_orientations is None:
      camera_orientations = self.camera_orientations

    (diffuse_light_visibility,
     specular_visib_outputs) = relit_lib.run_relit_preprocess(
         self._all_normals, hdri_map, camera_orientations)

    reflection_ratios = self._all_intrinsic
    specular_light_visibility_pre = relit_lib.combine_specular_light_visibilities(
        specular_visib_outputs[0], specular_visib_outputs[1],
        specular_visib_outputs[2], specular_visib_outputs[3], reflection_ratios)

    diffuse_specular_visibility = tf.concat(
        [diffuse_light_visibility, specular_light_visibility_pre], axis=-1)

    # Create images with NeRF
    (diffuse_specular_visibility, _,
     _) = volumetric_rendering.render_radiance_field(
         diffuse_specular_visibility,
         self._all_density,
         self._all_z_vals,
         sample_at_infinity=self._sample_at_infinity,
         clamp_mode=self._clamp_mode,
         noise_std=0.0)
    diffuse_specular_visibility = tf.reshape(
        diffuse_specular_visibility,
        (batch_size, self.nerf_resolution[0], self.nerf_resolution[1], 6))

    diffuse_specular_visibility = tf.image.resize(
        diffuse_specular_visibility,
        (self.high_resolution[0], self.high_resolution[1]), 'bilinear')

    (diffuse_light_visibility, specular_light_visibility_pre) = tf.split(
        diffuse_specular_visibility, [3, 3], axis=-1)

    diffuse_light_visibility = diffuse_light_visibility * tf.stop_gradient(
        input_mask) * tf.cast(input_mask > 0.1, tf.float32)
    specular_light_visibility_pre = specular_light_visibility_pre * tf.stop_gradient(
        input_mask) * tf.cast(input_mask > 0.1, tf.float32)

    specular_light_visibility = relit_lib.postpreocess_specular_visibility(
        specular_light_visibility_pre)

    high_image = rgb['high'][-1]
    relight_input = tf.concat(
        [high_image, diffuse_light_visibility, specular_light_visibility],
        axis=-1)
    relight_param = self.render_bottle_neck_mlps_3(relight_input)
    mul, res = tf.split(relight_param, [3, 3], axis=-1)
    relight = (high_image + 1.0) / 2.0 * (mul + 1) + res
    relight = relight * tf.cast(input_mask > 0.1, tf.float32)
    relight = relight * 2.0 - 1.0
    return (relight, diffuse_light_visibility, specular_light_visibility)

  def generate_avg_frequencies(
      self,
      num_sample: int = 1000
  ) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """Calculates average frequencies and phase shifts."""
    latent_code = tf.random.normal((num_sample, self.latent_dim))
    nerf_style_feature, conv_style_feature, _ = self.mapping_network(
        latent_code)

    for idx in range(len(nerf_style_feature)):
      nerf_style_feature[idx] = tf.reduce_mean(
          nerf_style_feature[idx], axis=0, keepdims=True)
    for idx in range(len(conv_style_feature)):
      conv_style_feature[idx] = tf.reduce_mean(
          conv_style_feature[idx], axis=0, keepdims=True)
    self.nerf_style_feature = nerf_style_feature
    self.conv_style_feature = conv_style_feature
    return self.nerf_style_feature, self.conv_style_feature

  def inference_interpolate(
      self,
      latent_code: tf.Tensor,
      img_size: Tuple[int, int] = (128, 128),
      fov: float = 13.0,
      ray_start: float = 0.0,
      ray_end: float = 2.0,
      num_samples: int = 16,
      h_stddev: float = 1.0,
      v_stddev: float = 1.0,
      h_mean: float = math.pi * 0.5,
      v_mean: float = math.pi * 0.5,
      psi: float = 1.0,
      sample_dist: Optional[str] = None,
      hierarchical_num_samples: bool = False,
      white_back: bool = False,
      has_background: bool = False,
      sample_at_infinity: bool = True,
      clamp_mode: str = 'relu',
      nerf_noise: float = 0.5,
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor],
             Dict[str, tf.Tensor], tf.Tensor]:
    """Similar to genereic forward but used for interpolated mapping embeddings."""
    batch_size = latent_code.shape.as_list()[0]
    self.generate_avg_frequencies()

    mapping_embeddings = self.mapping_network.inference_mapping_embeddings(
        latent_code)

    mapping_embeddings = self.mapping_network.w_avg[None, ...] + (
        mapping_embeddings - self.mapping_network.w_avg[None, ...]) * psi

    mapping_embeddings = self.mapping_network.broadcast(mapping_embeddings)

    (nerf_style_feature, conv_style_feature
    ) = self.mapping_network.inference_style_vectors(mapping_embeddings)

    # Generate initial camera rays and sample points.
    (points, z_vals, ray_directions, ray_local_directions, ray_origins, azimuth,
     elevation,
     world2camera_rotation) = ray_generator.get_ray_points_from_sampled_camera(
         batch_size,
         fov,
         img_size,
         h_stddev,
         v_stddev,
         h_mean,
         v_mean,
         num_samples,
         ray_start,
         ray_end,
         sample_dist,
         return_rotation_matrix=True)

    rgb, depths, normals, masks, angles = self.inference_from_style_embeddings(
        nerf_style_feature=nerf_style_feature,
        conv_style_feature=conv_style_feature,
        points=points,
        z_vals=z_vals,
        ray_directions=ray_directions,
        ray_local_directions=ray_local_directions,
        ray_origins=ray_origins,
        azimuth=azimuth,
        elevation=elevation,
        world2camera_rotation=world2camera_rotation,
        batch_size=batch_size,
        img_size=img_size,
        ray_start=ray_start,
        ray_end=ray_end,
        num_samples=num_samples,
        fov=fov,
        hierarchical_num_samples=hierarchical_num_samples,
        white_back=white_back,
        has_background=has_background,
        sample_at_infinity=sample_at_infinity,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
    )
    return rgb, depths, normals, masks, angles

  def inference_style_angle(
      self,
      nerf_style_feature: Sequence[tf.Tensor],
      conv_style_feature: Sequence[tf.Tensor],
      angles: tf.Tensor,
      img_size: Tuple[int, int] = (128, 128),
      fov: float = 13.0,
      ray_start: float = 0.0,
      ray_end: float = 2.0,
      num_samples: int = 16,
      hierarchical_num_samples: int = 32,
      white_back: bool = False,
      has_background: bool = False,
      sample_at_infinity: bool = True,
      clamp_mode: str = 'relu',
      nerf_noise: float = 0.5,
      return_depth: bool = False
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor],
             Dict[str, tf.Tensor], tf.Tensor]:
    """Similar to genereic forward but used for specified camera angle."""
    batch_size = angles.shape.as_list()[0]
    azimuth = angles[..., :1]
    elevation = angles[..., 1:]

    # Generate initial camera rays and sample points.
    (points, z_vals, ray_directions, ray_local_directions, ray_origins, azimuth,
     elevation, world2camera_rotation
    ) = ray_generator.get_ray_points_from_camera_with_angles(
        batch_size,
        azimuth,
        elevation,
        fov,
        img_size,
        num_samples=num_samples,
        ray_start=ray_start,
        ray_end=ray_end,
        return_rotation_matrix=True,
    )

    rgb, depths, normals, masks, angles = self.inference_from_style_embeddings(
        nerf_style_feature=nerf_style_feature,
        conv_style_feature=conv_style_feature,
        points=points,
        z_vals=z_vals,
        ray_directions=ray_directions,
        ray_local_directions=ray_local_directions,
        ray_origins=ray_origins,
        azimuth=azimuth,
        elevation=elevation,
        world2camera_rotation=world2camera_rotation,
        batch_size=batch_size,
        img_size=img_size,
        ray_start=ray_start,
        ray_end=ray_end,
        fov=fov,
        num_samples=num_samples,
        hierarchical_num_samples=hierarchical_num_samples,
        white_back=white_back,
        has_background=has_background,
        sample_at_infinity=sample_at_infinity,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
    )
    return rgb, depths, normals, masks, angles

  def inference_latent_code_angle(
      self,
      latent_code: tf.Tensor,
      angles: tf.Tensor,
      img_size: Tuple[int, int] = (128, 128),
      fov: float = 13.0,
      ray_start: float = 0.0,
      ray_end: float = 2.0,
      num_samples: int = 16,
      hierarchical_num_samples: int = 16,
      white_back: bool = False,
      has_background: bool = False,
      sample_at_infinity: bool = True,
      clamp_mode: str = 'rely',
      nerf_noise: float = 0.5,
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor],
             Dict[str, tf.Tensor], tf.Tensor]:
    """Similar to genereic forward but used for specified camera angle and latent code."""
    batch_size = latent_code.shape.as_list()[0]
    azimuth = angles[..., :1]
    elevation = angles[..., 1:]
    (nerf_style_feature, conv_style_feature,
     _) = self.mapping_network(latent_code)

    # Generate initial camera rays and sample points.
    (points, z_vals, ray_directions, ray_local_directions, ray_origins, azimuth,
     elevation, world2camera_rotation
    ) = ray_generator.get_ray_points_from_camera_with_angles(
        batch_size,
        azimuth,
        elevation,
        fov,
        img_size,
        num_samples=num_samples,
        ray_start=ray_start,
        ray_end=ray_end,
        return_rotation_matrix=True,
    )

    rgb, depths, normals, masks, angles = self.inference_from_style_embeddings(
        nerf_style_feature=nerf_style_feature,
        conv_style_feature=conv_style_feature,
        points=points,
        z_vals=z_vals,
        ray_directions=ray_directions,
        ray_local_directions=ray_local_directions,
        ray_origins=ray_origins,
        azimuth=azimuth,
        elevation=elevation,
        world2camera_rotation=world2camera_rotation,
        batch_size=batch_size,
        img_size=img_size,
        ray_start=ray_start,
        ray_end=ray_end,
        num_samples=num_samples,
        fov=fov,
        hierarchical_num_samples=hierarchical_num_samples,
        white_back=white_back,
        has_background=has_background,
        sample_at_infinity=sample_at_infinity,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
    )
    return rgb, depths, normals, masks, angles

  def mix_mapping_embeddings(self, latent_code1: tf.Tensor,
                             mapping_embeddings1: tf.Tensor) -> tf.Tensor:
    """Randomly mix mapping embeddings."""
    latent_code2 = tf.random.normal(
        shape=tf.shape(latent_code1), dtype=tf.dtypes.float32)
    mapping_embeddings2 = self.mapping_network.inference_mapping_embeddings(
        latent_code2)
    mapping_embeddings2 = self.mapping_network.broadcast(mapping_embeddings2)

    if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
      mixing_cutoff_index = tf.random.uniform(
          [], 5, self.mapping_network.total_broadcast, dtype=tf.dtypes.int32)
    else:
      mixing_cutoff_index = tf.constant(
          self.mapping_network.total_broadcast, dtype=tf.dtypes.int32)

    mixed_w_broadcasted = tf.where(
        condition=tf.broadcast_to(
            self.mixing_layer_indices < mixing_cutoff_index,
            tf.shape(mapping_embeddings1)),
        x=mapping_embeddings1,
        y=mapping_embeddings2)
    return mixed_w_broadcasted
