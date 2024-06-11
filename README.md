# U-RISC Cell Edge Detection

## Environment
### Hardware
- GPU (Apple M1 Pro,16-core)
- CPU(16 GB)

### Software
- macOS 14.0，Darwin 23.0.0
- Python 3.9.0

#### Steps
```bash
git clone git@github.com:007DXR/U-RISC-Cell-Edge-Detection.git
cd ./U-RISC-Cell-Edge-Detection
python3.9 -m virtualenv venv 
source ./venv/bin/activate 
pip install -r requirements.txt
```

Install datasets and models from [link](https://disk.pku.edu.cn/link/AA35E85994EA434DE88A1B27F148E78FB3). Put it in folder `/U-RISC-Cell-Edge-Detection`.

## Dataset
[U-RISC](https://www.biendata.xyz/competition/urisc/data/) is an annotated Ultra-high Resolution Image Segmentation dataset for cell membrane.

There are two tracks: the Simple one and the Complex one.

The Simple track has fewer cell counts, smaller image size, lower resolution and fewer pixels on the cell membrane. The Simple track has 30 training images, 9 validation images and 30 test images, the resolution of each image and label is 1024×1024.

The Complex track has more cell counts, larger image size, higher resolution and more pixels on the cell membrane.


Dataset is also available from [link](https://pan.baidu.com/s/1MD_ESaszcLv3xxQmJXPzfw#list/path=%2F). password: 20b8


## Model

Architecture model : [DFF](https://arxiv.org/abs/1902.09104)、[CASENet](https://arxiv.org/abs/1705.09759)

Backbone model : ResNet-152、ResNet-50

Models are also available from [link](https://pan.baidu.com/s/1Fmc9tb9AYQMb54sBpS66Dw). password: 8eqb


## Training
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset simple --model DFF --backbone resnet50 --batch-size 8 --lr 0.001 --epochs 200 --crop-size 960 --kernel-size 5 --edge-weight 0.4
```
#### Simple Track (Visualization)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vis.py --dataset simple --model DFF --backbone resnet50 --batch-size 8 --lr 0.001 --epochs 200 --crop-size 960 --kernel-size 5 --edge-weight 0.4
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset complex --model CASENet --backbone resnet152 --batch-size 4 --lr 0.001 --epochs 100 --crop-size 1280 --kernel-size 9 --edge-weight 0.4
```
#### Complex Track (Visualization)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vis.py --dataset complex --model CASENet --backbone resnet152 --batch-size 4 --lr 0.001 --epochs 100 --crop-size 1280 --kernel-size 9 --edge-weight 0.4
```

## Validaing and Morphological Processing
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --dataset simple --model DFF --backbone resnet50
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --dataset complex --model CASENet --backbone resnet152
```
#### Morphological Processing
```bash
python val_mor.py
```

## Testing and Ensembling
#### Simple Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset simple --model DFF --backbone resnet50
```
#### Complex Track
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset complex --model CASENet --backbone resnet152
```
