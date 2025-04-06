# !/bin/bash
set -e

# Create conda environment.
project_dir=$(pwd)
# Conda init
conda_home=$(conda info | grep "base environment" | awk '{print $4}')
source $conda_home/etc/profile.d/conda.sh

conda_env_name=gs_toolkit
conda create -n $conda_env_name python=3.8 -y

# Activate the environment
conda activate $conda_env_name

# Install CUDA toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install Gaussian Splatting Rasterizer
pip install -e ./dependencies/diff-gaussian-rasterization

# Install taichi_3d_ellipsoid
pip install -e ./dependencies/taichi_3d_ellipsoid

# Install requirements
pip install -r requirements.txt

# Install this repo as a package
pip install -e .