<div align="center">

<h1>GeoGS-SLAM: Online Monocular Reconstruction Using Gaussian Splatting with Geometric Priors</h1>

ICRA 2026

</div>

<div align="center">

[![Paper](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv)](https://rlgao.github.io/geogs_slam/)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://rlgao.github.io/geogs_slam/)

</div>

---

## Install

Refer to: environment.yml for conda env setup.

```bash
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
./setup.sh # or: do it step by step
```

## Checkpoints

Download model checkpoints from:

- [VGGT](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) 
- [MegaLoc](https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch) 
- [DINOv2 SALAD](https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt)

and put them under checkpoints/:

```
checkpoints/
├── dino_salad.ckpt
├── megaloc.torch
└── vggt_model_1B.pt
```

## Run

```bash
bash eval/run_replica.sh
# or: python run.py --config config/replica/office0.yaml --save_path_parent "results_tmp"
```

