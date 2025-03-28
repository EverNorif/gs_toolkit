import cv2
import os
import numpy as np

from glob import glob
from tqdm import tqdm
from typing import List, Tuple

from .general_utils import convert_to_camera_model, convert_gs_camera_to_cv2_camera
from ..model.camera_model import Camera


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def transform_poses_pca(c2ws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    positions = c2ws[:, :3, 3]
    center = positions.mean(axis=0)
    recentered_positions = positions - center

    eigvals, eigvecs = np.linalg.eig(recentered_positions.T @ recentered_positions)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]
    rotation = eigvecs.T
    # Make sure the coordinate system is right-handed.
    if np.linalg.det(rotation) < 0:
        rotation = np.diag(np.array([1, 1, -1])) @ rotation
    transform = np.concatenate([rotation, rotation @ -center[:, None]], -1) 
    poses_recentered = unpad_poses(transform @ pad_poses(c2ws))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_ellipse_path(poses: np.ndarray,
                         n_frames: int = 480,
                         z_variation: float = 0.,
                         z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                          (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])

def generate_bounding_trajectory(cameras: List[Camera], n_frames: int=480) -> List[Camera]:
    c2ws, K = [], None
    width, height = cameras[0].image_width, cameras[0].image_height
    for camera in cameras:
        c2w, K = convert_gs_camera_to_cv2_camera(camera)
        c2ws.append(c2w)
    pose_recenter, pca_transfrom = transform_poses_pca(np.array(c2ws))

    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    new_poses = np.linalg.inv(pca_transfrom) @ pad_poses(new_poses)
    new_cameras = []
    for new_c2w in new_poses:
        new_camera = convert_to_camera_model(c2w=new_c2w, fx=K[0, 0], fy=K[1, 1], width=width, height=height)
        new_cameras.append(new_camera)
    return new_cameras

def save_images_to_video(image_folder: str, output_path: str, fps: int=30):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_folder, ext)))
    image_files.sort()
    if not image_files:
        print("no images found")
        return
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))    
    for image_file in tqdm(image_files):
        frame = cv2.imread(image_file)
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"video saved to: {output_path}")

def save_c2ws_to_cameras(c2ws, output_path: str, fx, fy, width, height):
    from gs_toolkit.utils.general_utils import save_cameras_json, convert_to_camera_model
    cameras = []
    for c2w in c2ws:
        camera = convert_to_camera_model(c2w=c2w, fx=fx, fy=fy, width=width, height=height)
        cameras.append(camera)
    save_cameras_json(cameras, output_path)
