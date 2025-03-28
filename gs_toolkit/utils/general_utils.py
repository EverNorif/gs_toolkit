import os
import json
import torch
import numpy as np

from typing import List

from .graphics_utils import focal2fov, fov2focal, getWorld2View2
from ..model.camera_model import Camera


def inverse_sigmoid(x):
    return torch.log(x/(1-x))
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_extrinsic(rotation, translation):
    if isinstance(translation, list):
        translation = np.array(translation)
    if isinstance(rotation, list):
        rotation = np.array(rotation)
    translation = translation.reshape(3, 1)
    extrinsic = np.concatenate([rotation, translation], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
    return extrinsic

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if fname.startswith("iteration_")]
    return max(saved_iters)

def convert_gs_camera_to_cv2_camera(camera: Camera):
    w2c = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
    c2w = np.linalg.inv(w2c)
    extrinsic = c2w

    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx, cy = camera.image_width / 2, camera.image_height / 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    return extrinsic, intrinsic

def convert_to_camera_model(c2w, fx, fy, width, height):
    w2c = np.linalg.inv(c2w)
    t, R = w2c[:3, 3], w2c[:3, :3]
    R = R.transpose() # KEY_POINT: R is stored transposed due to 'glm' in CUDA code
    return Camera(
        width=width,
        height=height,
        FoVx=focal2fov(fx, width),
        FoVy=focal2fov(fy, height),
        R=R,
        T=t
    )

def parse_cameras_json(cameras_json_path: str) -> List[Camera]:
    with open(cameras_json_path, 'r') as f:
        cameras_info = json.load(f)
    cameras = []
    for camera_info in cameras_info:
        width, height = int(camera_info['width']), int(camera_info['height'])
        rotation, translation = camera_info['rotation'], camera_info['position']
        fx, fy = camera_info['fx'], camera_info['fy']
        c2w = build_extrinsic(rotation, translation)
        camera = convert_to_camera_model(c2w, fx, fy, width, height)
        cameras.append(camera)
    return cameras

def save_cameras_json(cameras: List[Camera], output_path: str):
    cameras_info = []
    for idx, camera in enumerate(cameras):
        w2c = getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)
        c2w = np.linalg.inv(w2c)
        rotation, translation = c2w[:3, :3], c2w[:3, 3]
        cameras_info.append({
            'id': idx,
            'img_name': f"{idx:05d}",
            'width': camera.image_width,
            'height': camera.image_height,
            'rotation': rotation.tolist(),
            'position': translation.tolist(),
            'fx': fov2focal(camera.FoVx, camera.image_width),
            'fy': fov2focal(camera.FoVy, camera.image_height),
        })
    with open(output_path, 'w') as f:
        json.dump(cameras_info, f)
