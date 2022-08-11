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

"""This module glues different parts of the model together."""
import math

import gin
import tensorflow as tf

from generator import relit_generator

@gin.configurable()
def create_models(
    latent_dim=256,
    num_samples=12,
    hierarchical_num_samples=12,
    image_size=(64, 64),
    holodeck_image_size=(64, 64),
    high_image_size=(128, 128),
    fov=13.0,
    ray_start=0.88,
    ray_end=1.12,
    h_stddev=0.3,
    v_stddev=0.155,
    h_depth_stddev=0.3,
    v_depth_stddev=0.155,
    h_mean=math.pi * 0.5,
    v_mean=math.pi * 0.5,
    sample_dist='gaussian',
    depth_sample_dist='truncated_gaussian',
    clamp_mode='relu',
    latent_code_dist='gaussian',
    has_background=False,
    sample_at_infinity=True,
    white_back=False,
    device='gpu',
):
  """Initializes Nerf-related models based on the configuration."""
  generator = relit_generator.ImplicitGenerator3d(
    latent_dim=latent_dim, upsample_factor=4, device=device)

  models = {
      'generator': generator,
  }

  model_kwargs = {
      'latent_dim': latent_dim,
      'num_samples': num_samples,
      'hierarchical_num_samples': hierarchical_num_samples,
      'image_size': image_size,
      'holodeck_image_size': holodeck_image_size,
      'high_image_size': high_image_size,
      'fov': fov,
      'ray_start': ray_start,
      'ray_end': ray_end,
      'h_stddev': h_stddev,
      'v_stddev': v_stddev,
      'h_depth_stddev': h_depth_stddev,
      'v_depth_stddev': v_depth_stddev,
      'h_mean': h_mean,
      'v_mean': v_mean,
      'sample_dist': sample_dist,
      'depth_sample_dist': depth_sample_dist,
      'clamp_mode': clamp_mode,
      'latent_code_dist': latent_code_dist,
      'has_background': has_background,
      'sample_at_infinity': sample_at_infinity,
      'white_back': white_back,
  }

  return models, model_kwargs


@gin.configurable()
def create_train_params(
    topk_interval=2000.0,
    topk_v=0.75,
    fade_steps=5000.0,
    latent_code_dist='gaussian',
    loss_weight={
        'r_color_lambda': 0.2,
        'r_depth_lambda': 200.0,
        'r_scalar_lambda': 1.0,
        'direction_reg': 5.0,
        'pos_lambda': 15.0,
        'patch_weight': 0.5,
        'relit_weight': 1.0,
    },
):
  train_params = {
      'topk_interval': topk_interval,
      'topk_v': topk_v,
      'fade_steps': fade_steps,
      'latent_code_dist': latent_code_dist,
      'loss_weight': loss_weight,
  }
  return train_params
