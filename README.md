# GeoGS-SLAM: Online Monocular Reconstruction Using Gaussian Splatting with Geometric Priors

## Install
```
# refer to: vggt_gs_recon.yml

cd GeoGS_SLAM
conda create -n geogs_slam python=3.12
conda activate geogs_slam

# on-the-fly-nvs
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
pip install -r requirements.txt

# vggt_slam
cd vggt_slam
chmod +x setup.sh
./setup.sh
```