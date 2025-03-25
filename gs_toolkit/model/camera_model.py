import torch
import numpy as np
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera:
    def __init__(self, width, height, FoVx, FoVy, R, T, znear=0.01, zfar=100.0, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.image_width = width
        self.image_height = height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.znear = znear
        self.zfar = zfar
        self.R = R # R in w2c
        self.T = T # T in w2c
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  # w2c
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]