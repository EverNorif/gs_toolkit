# gs_toolkit

This repository provides some useful tools for [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/EverNorif/gs_toolkit.git
cd gs_toolkit

# Create and activate conda environment
bash setup.sh
```

## Data Preparation

We use the output results of [vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) as our input. You can run the official repo to get the reconstruction data. You can also use the reconstruction result of the bicycle scene from [MipNeRF-360](https://jonbarron.info/mipnerf360/) as an example. Download it [here](https://drive.google.com/drive/folders/1mxp4tr8YtZL4tVK8k0MC3CEmgvmbezat?usp=sharing).

```shell
data/
├── bicycle/
│ ├── cameras.json
│ ├── point_cloud/
│ │ ├── iteration_xxx/
│ │ │ └── point_cloud.ply
│ │ └── .../
│ └── ...
└── ...
```

## Usage

We provide both Python and shell interfaces for running the toolkit.

### Camera Rendering

Camera Rendering will render images at each camera position defined in `cameras.json`.

- python

```python
from gs_toolkit.entry import Entry

entry = Entry(scene_path='./data/bicycle', output_path='./output')
entry.render_cameras()
```

- bash

```shell


```

### Video Rendering

Video Rendering will render a surrounding video of the scene like the following result.

<details>
<summary>[Video Rendering Example]</summary>
<p></p>
(Note: Demo video is compressed. The actual rendered videos will be at full resolution)
<p></p>
  
https://github.com/user-attachments/assets/f92006ff-1122-478a-9528-ce23810a17f1

</details>

- python

```python
from gs_toolkit.entry import Entry

entry = Entry(scene_path='./data/bicycle', output_path='./output')
entry.render_video()
```

- shell

```shell


```

## Additional Information

### Camera Conventions

There are some differences between OpenCV camera and the [Camera](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/cameras.py) used in the official 3DGS implementation. Here are some notes:

- The rotation and translation in `cameras.json` are in camera-to-world (c2w) format.

- `getWorld2View2` can get the world-to-camera (w2c) transformation for OpenCV cameras, which can be inverted to get camera-to-world (c2w).

- `gs_toolkit.utils.general_utils.convert_gs_camera_to_cv2_camera` and `gs_toolkit.utils.general_utils.convert_to_camera_model`
provide conversion methods between OpenCV cameras and 3DGS cameras.

## Acknowledge

Some code in this repo is borrowed from [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), thanks their great work.
