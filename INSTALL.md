# VecHeart Installation

The below installation process is verified on NVIDIA A100 + PyTorch 2.8.0 + CUDA 12.6. If you have different GPU or CUDA version, change them accordingly.

## 1. Create the conda environment
```bash
conda create -n vecheart python=3.10 -y
conda activate vecheart
```

## 2. Install PyTorch and the version-coupled 3D libraries
kaolin and PyTorch3D ship prebuilt wheels tied to the exact PyTorch + CUDA version. If you change the torch version, adjust the versions/URLs below accordingly (see the kaolin and PyTorch3D wheel indices).
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html
pip install pytorch3d==0.7.9+pt2.8.0cu126 --extra-index-url https://miropsota.github.io/torch_packages_builder
# FlashAttention is also compiled against the exact torch/CUDA version
pip install flash-attn --no-build-isolation
```

## 3. Install reimukit
Place the `reimukit/` directory next to this repo, then:
```bash
git clone https://github.com/Scalsol/reimukit.git

cd reimukit
pip install -e .
cd ..
```

## 4. Install VecHeart
The remaining Python dependencies are declared in `setup.py` (`install_requires`), This command installs them together with VecHeart.
```bash
git clone https://github.com/Scalsol/VecHeart.git

cd VecHeart
pip install -e .
```

## Verifying the install
```bash
# torch-coupled 3D libs (after step 2)
python -c "import kaolin; from kaolin.ops.conversions import marching_tetrahedra; print(kaolin.__version__)"
python -c "from pytorch3d.loss.chamfer import chamfer_distance; from pytorch3d.structures import Meshes"
python -c "from flash_attn import flash_attn_func"
# reimukit and the deps it provides (after step 3)
python -c "import reimu, scipy, cv2, wandb"
# vecheart and its remaining deps (after step 4)
python -c "import vecheart, torchmetrics, sklearn, einops, mcubes, trimesh"
```
