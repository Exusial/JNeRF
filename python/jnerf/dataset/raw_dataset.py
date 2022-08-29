import random
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

import jittor as jt
from jittor.dataset import Dataset
import os
import json
import cv2
import imageio
from math import pi
from math import tan
from tqdm import tqdm
import numpy as np
from jnerf.utils.registry import DATASETS
from jnerf.utils.miputils import *
from .dataset_util import *
from jnerf.utils import camera_utils, rawutils
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.dataset.dataset import MipNerfDataset

# TODO: add pycolmal from JAX.
import sys
sys.path.insert(0,'python/jnerf/dataset/pycolmap')
sys.path.insert(0,'python/jnerf/dataset/pycolmap/pycolmap')
import pycolmap

class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

  def process(
      self
  ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, Optional[Mapping[
      Text, float]], camera_utils.ProjectionType]:
    """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

    self.load_cameras()
    self.load_images()
    # self.load_points3D()  # For now, we do not need the point cloud data.

    # Assume shared intrinsics between all cameras.
    cam = self.cameras[1]

    # Extract focal lengths and principal point parameters.
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

    # Extract extrinsic matrices in world-to-camera format.
    imdata = self.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
      params = None
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 1 or type_ == 'PINHOLE':
      params = None
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    if type_ == 2 or type_ == 'SIMPLE_RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 3 or type_ == 'RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 4 or type_ == 'OPENCV':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['p1'] = cam.p1
      params['p2'] = cam.p2
      camtype = camera_utils.ProjectionType.PERSPECTIVE

    elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['k3'] = cam.k3
      params['k4'] = cam.k4
      camtype = camera_utils.ProjectionType.FISHEYE

    return names, poses, pixtocam, params, camtype

@DATASETS.register_module()
class RawLLFF(MipNerfDataset):
    def __init__(self,root_dir, batch_size, mode='train', H=0, W=0, near=0., far=1., correct_pose=[1,-1,-1], aabb_scale=None, offset=None, img_alpha=True, have_img=True, preload_shuffle=True):
        super().__init__(root_dir, batch_size, mode, H, W, near, far, correct_pose, aabb_scale, offset, img_alpha, have_img, preload_shuffle, True) # not load data in previous way.
        self.cfg = get_cfg()
        self._load_renderings()
        jt.gc()


    def _load_renderings(self):
        """Load images from disk."""
        # Set up scaling factor.
        image_dir_suffix = ''
        # Use downsampling factor (unless loading training split for raw dataset,
        # we train raw at full resolution because of the Bayer mosaic pattern).
        if self.cfg.factor > 0 and not (self.rawnerf_mode and
                                    self.mode == 'train'):
            image_dir_suffix = f'_{self.cfg.factor}'
            factor = self.cfg.factor
        else:
            factor = 1

        # Copy COLMAP data to local disk for faster loading.
        colmap_dir = os.path.join(self.root_dir, 'sparse/0/')

        # Load poses.
        if os.path.exists(colmap_dir):
            pose_data = NeRFSceneManager(colmap_dir).process()
        else:
            # Attempt to load Blender/NGP format if COLMAP data not present.
            pose_data = load_blender_posedata(self.root_dir)
        image_names, poses, pixtocam, distortion_params, camtype = pose_data

        # Previous NeRF results were generated with images sorted by filename,
        # use this flag to ensure metrics are reported on the same test set.
        # if self.cfg.load_alphabetical:
        #     inds = np.argsort(image_names)
        #     image_names = [image_names[i] for i in inds]
        #     poses = poses[inds]

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocam = pixtocam @ np.diag([factor, factor, 1.])
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal_lengths = [1. / self.pixtocams[0, 0], 1. / self.pixtocams[0, 0]]
        self.distortion_params = distortion_params
        self.camtype = camtype

        raw_testscene = False
        if self.rawnerf_mode:
        # Load raw images and metadata.
            images, metadata, raw_testscene = rawutils.load_raw_dataset(
                self.mode, # TODO: FIX SPLIT
                self.root_dir,
                image_names,
                self.cfg.exposure_percentile,
                factor)
            self.metadata = metadata
        print("finish load.")
        # Load bounds if possible (only used in forward facing scenes).
        posefile = os.path.join(self.root_dir, 'poses_bounds.npy')
        if os.path.exists(posefile):
            with open(posefile, 'rb') as fp:
                poses_arr = np.load(fp)
            bounds = poses_arr[:, -2:]
        else:
            bounds = np.array([0.01, 1.])
        self.colmap_to_world_transform = np.eye(4)

        # Separate out 360 versus forward facing scenes.
        if self.cfg.forward_facing:
            # Set the projective matrix defining the NDC transformation.
            self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
            # Rescale according to a default bd factor.
            scale = 1. / (bounds.min() * .75)
            poses[:, :3, 3] *= scale
            self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
            bounds *= scale
            # Recenter poses.
            poses, transform = camera_utils.recenter_poses(poses)
            self.colmap_to_world_transform = (
                transform @ self.colmap_to_world_transform)
            # Forward-facing spiral render path.
            self.render_poses = camera_utils.generate_spiral_path(
                poses, bounds, n_frames=self.cfg.render_path_frames)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform = camera_utils.transform_poses_pca(poses)
            self.colmap_to_world_transform = transform
            if self.cfg.render_spline_keyframes is not None:
                rets = camera_utils.create_render_spline_path(self.cfg, image_names,poses, self.exposures)
                self.spline_indices, self.render_poses, self.render_exposures = rets
            else:
                # Automatically generated inward-facing elliptical render path.
                self.render_poses = camera_utils.generate_ellipse_path(
                    poses,
                    n_frames=self.cfg.render_path_frames,
                    z_variation=self.cfg.z_variation,
                    z_phase=self.cfg.z_phase)

        if raw_testscene:
            # For raw testscene, the first image sent to COLMAP has the same pose as
            # the ground truth test image. The remaining images form the training set.
            raw_testscene_poses = {
                'test': poses[:1],
                'train': poses[1:],
            }
            poses = raw_testscene_poses[self.mode]
        poses = poses[:2]
        self.poses = poses

        # Select the split.
        all_indices = np.arange(images.shape[0])
        if self.cfg.llff_use_all_images_for_training or raw_testscene:
            train_indices = all_indices
        else:
            train_indices = all_indices % self.cfg.llffhold != 0
        split_indices = {
            'test': all_indices[all_indices % self.cfg.llffhold == 0],
            'train': all_indices[train_indices],
        }
        print("split_indices: ", split_indices)
        indices = split_indices[self.mode]
        # All per-image quantities must be re-indexed using the split indices.
        if self.preload_shuffle:
            indices = indices[jt.randperm(indices.shape[0]).numpy()]
        images = images[indices]
        poses = poses[indices]
        if self.exposures is not None:
            self.exposures = self.exposures[indices]
        if self.rawnerf_mode:
            for key in ['exposure_idx', 'exposure_values']:
                self.metadata[key] = self.metadata[key][indices]
        # TODO: simtaneously generate train/test dataset in this function
        self.n_images = images.shape[0]
        self.image_data = images
        # compared to transform gpu
        self.transforms_gpu = self.render_poses if self.cfg.render_path else poses
        self.transforms_gpu = jt.array(self.transforms_gpu)[indices]
        self.H, self.W = images.shape[1:3]
        self.img_ids = jt.array(np.arange(self.n_images)[None,...].reshape(self.n_images, 1).repeat(self.H*self.W, 1))
        self.img_ids = self.img_ids.reshape(self.n_images, self.H, self.W).unsqueeze(-1)
        self._generate_rays()
        print(list(map(lambda r: r.shape, self.rays)))
        self.rays = namedtuple_map(lambda r: r.reshape(self.n_images*self.H*self.W, -1), self.rays)