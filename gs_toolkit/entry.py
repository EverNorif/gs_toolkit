import os
import fire

import torch
import torchvision
from typing import List, Callable

from .model.camera_model import Camera
from .model.gaussian_model import GaussianModel
from .render.gaussian_render import render, render_ellipsoid, build_ellipsoid_renderer
from .utils.general_utils import searchForMaxIteration, parse_cameras_json
from .utils.render_utils import generate_bounding_trajectory, save_images_to_video
from .utils.rich_utils import get_progress, get_status, CONSOLE

class Entry:
    def __init__(self, scene_path: str, output_path: str, bg_color: torch.Tensor=[1.0, 1.0, 1.0]):
        self.scene_path = scene_path
        self.camera_path = os.path.join(scene_path, "cameras.json")

        ply_dir = os.path.join(scene_path, "point_cloud")
        max_iteration = searchForMaxIteration(ply_dir)
        self.ply_path = os.path.join(ply_dir, f"iteration_{max_iteration}", "point_cloud.ply")

        scene_name = os.path.basename(scene_path)
        self.output_path = os.path.join(output_path, scene_name)
        os.makedirs(self.output_path, exist_ok=True)

        if isinstance(bg_color, list):
            self.bg_color = torch.tensor(bg_color).cuda()
        else:
            self.bg_color = bg_color.cuda()
        self.gs_model = self.__load_gs_model()
        self.cameras = parse_cameras_json(self.camera_path)

    def render_cameras(self):
        save_dir = os.path.join(self.output_path, "renders")
        os.makedirs(save_dir, exist_ok=True)
        self.__save_render_results(cameras=self.cameras, save_dir=save_dir, msg=":camera_with_flash: Rendering cameras")
        CONSOLE.print("[bright_green]:party_popper: Rendering cameras has been saved to: ", save_dir)

    def render_video(self, frame_nums: int=480, fps: int=30):
        save_dir = os.path.join(self.output_path, "trajectory")
        os.makedirs(save_dir, exist_ok=True)

        cameras_trajectory = generate_bounding_trajectory(self.cameras, frame_nums)
        self.__save_render_results(cameras=cameras_trajectory, save_dir=save_dir, msg=":ringed_planet: Rendering trajectory")
        save_images_to_video(image_folder=save_dir, output_path=os.path.join(self.output_path, "trajectory.mp4"), fps=fps)
    
    def render_ellipsoids(self):
        save_dir = os.path.join(self.output_path, "ellipsoid_render")
        os.makedirs(save_dir, exist_ok=True)
        self.__save_ellipsoid_results(cameras=self.cameras, save_dir=save_dir, msg=":passenger_ship: Rendering ellipsoids")
        CONSOLE.print("[bright_green]:party_popper: Rendering ellipsoids has been saved to: ", save_dir)
    
    def render_ellipsoids_video(self, frame_nums: int=480, fps: int=30):
        save_dir = os.path.join(self.output_path, "ellipsoid_trajectory")
        os.makedirs(save_dir, exist_ok=True)

        cameras_trajectory = generate_bounding_trajectory(self.cameras, frame_nums)
        self.__save_ellipsoid_results(cameras=cameras_trajectory, save_dir=save_dir, msg=":penguin: Rendering trajectory with ellipsoids")
        save_images_to_video(image_folder=save_dir, output_path=os.path.join(self.output_path, "ellipsoid_trajectory.mp4"), fps=fps)

    def __load_gs_model(self) -> GaussianModel:
        CONSOLE.print(f"[bright_green]:cloud_with_tornado: Load gs model from {self.ply_path}")
        with get_status(f"[bright_green] Loading model...") as status:
            gs_model = GaussianModel()
            gs_model.load_ply(self.ply_path)
        return gs_model
    
    def __save_render_results(self, cameras: List[Camera], save_dir: str, msg: str):
        with get_progress(description=f"[bright_green]{msg}...", suffix="frame/s") as progress:
            for i, camera in enumerate(progress.track(cameras, total=len(cameras))):
                render_res = render(viewpoint_camera=camera, pc=self.gs_model, bg_color=self.bg_color)
                save_path = os.path.join(save_dir, f"{i:05d}.png")
                torchvision.utils.save_image(render_res['render'], save_path)
                torch.cuda.empty_cache()

    def __save_ellipsoid_results(self, cameras: List[Camera], save_dir: str, msg: str):
        ellipsoid_renderer = build_ellipsoid_renderer(viewpoint_camera=cameras[0], pc=self.gs_model, bg_color=self.bg_color)
        with get_progress(description=f"[bright_green]{msg}...", suffix="frame/s") as progress:
            for i, camera in enumerate(progress.track(cameras, total=len(cameras))):
                render_res = render_ellipsoid(viewpoint_camera=camera, renderer=ellipsoid_renderer)
                save_path = os.path.join(save_dir, f"{i:05d}.png")
                torchvision.utils.save_image(render_res, save_path)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    fire.Fire(Entry)
