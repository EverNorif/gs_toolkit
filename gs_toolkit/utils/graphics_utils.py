import torch
import math
import numpy as np

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def sh_features_to_colors(sh_features):
    C0 = 0.28209479177387814
    colors = sh_features * C0 + 0.5
    return colors

def quaternion_to_rotation_matrix(quaternions):
    assert quaternions.shape[1] == 4, "quaternions must be [N, 4] shape"
    
    norm = torch.sqrt(quaternions[:, 0]**2 + quaternions[:, 1]**2 + 
                     quaternions[:, 2]**2 + quaternions[:, 3]**2)
    quaternions = quaternions / norm.unsqueeze(1)
    
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    
    batch_size = quaternions.shape[0]
    rotmats = torch.zeros((batch_size, 3, 3), device=quaternions.device)
    
    rotmats[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rotmats[:, 0, 1] = 2 * (x*y - w*z)
    rotmats[:, 0, 2] = 2 * (x*z + w*y)
    
    rotmats[:, 1, 0] = 2 * (x*y + w*z)
    rotmats[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rotmats[:, 1, 2] = 2 * (y*z - w*x)
    
    rotmats[:, 2, 0] = 2 * (x*z - w*y)
    rotmats[:, 2, 1] = 2 * (y*z + w*x)
    rotmats[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return rotmats