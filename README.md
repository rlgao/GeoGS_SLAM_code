# GeoGS-SLAM: Online Monocular Reconstruction Using Gaussian Splatting with Geometric Priors

## Install
```
# refer to: vggt_gs_recon.yml

cd GeoGS_SLAM/
conda create -n geogs_slam python=3.12
conda activate geogs_slam

# on-the-fly-nvs
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x
pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/fused-ssim --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install submodules/graphdecoviewer

# vggt_slam
cd vggt_slam/

chmod +x setup.sh
./setup.sh
# [or: do it step by step]
```

## Run
```
bash eval/run_replica.sh
[or: python run.py --config config/replica/office0.yaml --save_path_parent "results_tmp"]
```
