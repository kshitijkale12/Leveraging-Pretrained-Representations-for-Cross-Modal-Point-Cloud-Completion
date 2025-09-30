# Leveraging Pretrained Representations for Cross-Modal Point Cloud Completion

This repository provides the official PyTorch implementation for our paper, **"Leveraging Pretrained Representations for Cross-Modal Point Cloud Completion"**. Our work introduces a novel image encoder that significantly improves the performance of the state-of-the-art EGIInet framework.

[](https://opensource.org/licenses/MIT)

*Our Dual-branch Encoder fuses geometric cues from a depth estimator with semantic priors from a Vision Transformer, creating a powerful guide for the EGIInet completion framework.*

-----

## Introduction

Image-guided point cloud completion aims to reconstruct complete 3D shapes from partial scans by leveraging a corresponding 2D image. However, training cross-modal networks from scratch often fails to capture the robust priors needed for challenging cases.

We challenge this paradigm by demonstrating that knowledge from large-scale, pre-trained models can be effectively transferred to this task. Our key contribution is a **Dual-branch Image Encoder** that fuses:

1.  **Geometric Cues:** Extracted from a pre-trained depth estimation model (DepthAnything).
2.  **Semantic Priors:** Extracted from a pre-trained Vision Transformer (DINOv2).

By replacing the original image encoder in the EGIInet framework with our module, we achieve a **7% performance increase** on unseen categories without altering the rest of the architecture, producing more semantically coherent and structurally accurate 3D shapes.

-----

## Setup and Installation

### 1\. Environment Setup

The code has been tested on `Ubuntu 20.04` with `Python 3.10`, `PyTorch 2.1.7`, and `CUDA 12.6`.

  * **Create a Conda environment (recommended):**

    ```bash
    conda create -n dinodepth python=3.10
    conda activate dinodepth
    ```

  * **Install PyTorch:**

    ```bash
    # Check https://pytorch.org/ for the command matching your system's CUDA version
    pip install torch torchvision torchaudio
    ```

  * **Install dependencies:**

    ```bash
    pip install easydict opencv-python transform3d h5py timm open3d tensorboardX ninja==1.11.1 torch-scatter einops
    ```

  * **Compile custom CUDA extensions:**

    ```bash
    # PointNet++ Utils
    cd models/pointnet2_batch && python setup.py install && cd ../..

    # Chamfer Distance
    cd metrics/CD/chamfer3D/ && python setup.py install && cd ../../..

    # Earth Mover's Distance
    cd metrics/EMD/ && python setup.py install && cd ../..

    # Furthest Point Sampling
    cd utils/furthestPointSampling/ && python setup.py install && cd ../..
    ```

### 2\. Dataset

Download the **ShapeNet-ViPC** dataset.

  * Visit the download link: [ShapeNetViPC-Dataset](https://www.google.com/search?q=https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH2A) (143GB, Access code: `ar8l`).
  * After downloading the compressed parts (`.tar.gz*`), combine and extract them:
    ```bash
    cat ShapeNetViPC-Dataset.tar.gz* | tar -zxvf -
    ```

This will create a `ShapeNetViPC-Dataset` directory containing the partial scans, ground truth point clouds, and viewpoint images.

### 3\. Pre-trained Models

Download the necessary pre-trained model weights from the link below and place them in your desired checkpoint directory.

  * **Download:** [google drive](https://drive.google.com/drive/folders/1Hej51WsV77XcqhQqW5BYDyUqFNS7JndN?usp=drive_link)

-----

## Usage

### Evaluation

To evaluate a pre-trained model, specify the path to your checkpoint in `config_vipc.py` and run the test script.

  * **Set the checkpoint path in `config_vipc.py`:**
    ```python
    __C.CONST.WEIGHTS = "path/to/your/checkpoint.pth"
    ```
  * **Run evaluation:**
    ```bash
    python main.py --test
    ```

### Training

To train the model from scratch, run:

```bash
python main.py
```

Training configurations can be adjusted in `config_vipc.py`.

-----

## Acknowledgement

This project is built upon the official implementation of **EGIINet**. Our code also references and utilizes components from the following outstanding repositories:

  * [EGIInet](https://github.com/WHU-USI3DV/EGIInet)
  * [Meta-Transformer](https://github.com/invictus717/MetaTransformer)
  * [ViPC](https://github.com/Hydrogenion/ViPC)
  * [XMFnet](https://github.com/diegovalsesia/XMFnet)

We thank all the authors for their valuable contributions and for open-sourcing their work..
