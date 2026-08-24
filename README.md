# HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone
*This is the official repository for the paper "HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone".* 
https://arxiv.org/abs/2605.17286

and **Hyperspectral Images Reader with multiple dataset support.**

[![BMVC 2026](https://img.shields.io/badge/BMVC-2026-blue.svg)](https://bmvc2026.bmva.org/)
[![Paper Status](<https://img.shields.io/badge/Status-Accepted%20to%20BMVC%202026-success.svg>)](https://bmvc2026.bmva.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.17286-b31b1b.svg)](https://arxiv.org/abs/2605.17286)
[![GitHub](https://img.shields.io/badge/GitHub-HyperVision-green.svg)](https://github.com/lronkitty/HyperVision)

> 📢 **Announcement**: Our paper **"HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone"** has been officially accepted by **BMVC 2026**! 🎉

*Official PyTorch implementation for the BMVC 2026 paper:*
**"HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone"**

---

## 🚀 News

- **[31-Jul-2026]** 🎉 **HyperVision** has been officially accepted to **BMVC 2026**!
- **[02-Jul-2026]** 📦 Official codebase, model checkpoints, and inference pipelines released.

---

## 📖 Abstract

While hyperspectral imaging provides rich spatial-spectral information across hundreds of narrow wavelength bands for precise material identification, ground-based hyperspectral pre-trained backbones remain absent, constrained by varying spectral configurations across sensors, limited annotations and heterogeneous labeling schemes, and the limited scale and scene diversity of existing datasets. To address these challenges and enable universal perception, we propose HyperVision, the first ground-based hyperspectral pre-trained backbone. First, to handle varying spectral configurations, HyperVision adopts a channel-adaptive dynamic embedding mechanism to map heterogeneous inputs into a unified token space. Second, we develop an unsupervised representation learning framework. Specifically, to address limited annotations and heterogeneous labeling schemes, a multi-source pseudo-labeling method is introduced to fuse spatial structures from SAM2 and fine-grained spectral material information from HyperFree. Furthermore, to enrich scene diversity and compensate for limited dataset scale, a cross-modal knowledge distillation mechanism is utilized to transfer rich semantic representations from a pre-trained RGB vision model to our backbone. Pre-trained on a collection of 15k images from 26 diverse ground-based datasets, HyperVision demonstrates exceptional generalization. Requiring only efficient head-only adaptation without adjusting backbone parameters, it outperforms state-of-the-art task-specific methods across three downstream tasks under varying sensor configurations, yielding up to a 16.3% relative improvement in hyperspectral semantic segmentation $\mathrm{Acc}_{\mathrm{M}}$, a 2.1% relative gain in object tracking AUC, and a 35.5% reduction in salient object detection MAE.

---

## 🛠️ Main Code (HyperVision)

The core architecture and model initialization utilities are located under the [HyperVision](./HyperVision) directory.

To initialize the model, you can use the model builder functions from [HyperVision/build_HyperVision.py](./HyperVision/build_HyperVision.py). We support different model configurations:

- **HyperVision-B**: `build_HyperVision_b`
- **HyperVision-L**: `build_HyperVision_l`
- **HyperVision-H** (Default): `build_HyperVision_h`

### Python Example:

```python
import torch
from HyperVision import build_HyperVision_h, build_HyperVision_l, build_HyperVision_b
from read_dataset_image import load_hypervision_matrix
from configs.hypervision.hyperspectral_pipelines import LoadHyperspectralImage

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 512

# 1. Build the HyperVision model (HyperVision-H configuration shown below)
model = build_HyperVision_h(
    checkpoint="path/to/hypervision_distilled_h.pth",             # Optional checkpoint path (.pth weights)
    image_size=512,                                         # Input resolution
    vit_patch_size=16,                                      # Patch size of the backbone
    encoder_global_attn_indexes=[15, 23, 31],               # Layers using global attention
    merge_indexs=[8, 32],                                   # Layers doing patch merging
    class_number=-1                                         # Number of classes for mask decoder
)
model = model.to(device)
model.eval()

# 2. Load and preprocess a hyperspectral image
# Preprocessed shape: (H_ori, W_ori, C_hsi) with values in range [0, 255]
img_matrix = load_hypervision_matrix("path/to/image.mat", dataset_name="harvard")
h_ori, w_ori = img_matrix.shape[:2]

# Convert to torch tensor with shape (1, C_hsi, H_ori, W_ori)
input_tensor = torch.as_tensor(img_matrix, device=device).permute(2, 0, 1).unsqueeze(0).float()

# 3. Automatically retrieve the wavelengths list from the dataset pipeline config
loader = LoadHyperspectralImage(dataset_type="harvard")
wavelengths = loader.wavelengths

# 4. Extract features via the HyperVision Image Encoder
with torch.no_grad():
    # GSD represents Ground Sampling Distance (defaults to [0.01])
    multi_stage_features = model.image_encoder(
        input_tensor, 
        test_mode=True, 
        input_wavelength=wavelengths, 
        GSD=[0.01]
    )
```

---

## 📊 Dataset Reading & Processing

We provide [read_dataset_image.py](./read_dataset_image.py) to read and preprocess hyperspectral images from various datasets. The script utilizes the custom data pipeline class `LoadHyperspectralImage` from [configs/hypervision/hyperspectral_pipelines.py](./configs/hypervision/hyperspectral_pipelines.py) to normalize the images (scaled to `[0, 255]`) and reshape them into the format expected by HyperVision: `(H_ori, W_ori, C_hsi)`.

### Python API Usage:

```python
from read_dataset_image import load_hypervision_matrix

# Load and preprocess HSI matrix
# dataset_name must be one of the supported dataset identifiers (e.g., 'harvard', 'arad_1k_31')
img_matrix = load_hypervision_matrix("path/to/image.mat", dataset_name="harvard")
print("Processed shape (H, W, C):", img_matrix.shape)
```

### Command Line Interface:

You can also run the script from the command line to load an HSI image and optionally save the processed matrix as a `.npy` file:

```bash
python read_dataset_image.py --path /path/to/image.mat --dataset harvard --output output.npy
```

---

## 🗃️ Supported Datasets

Below is a summary of the 26 ground-based hyperspectral datasets (31 subset entries) supported by our pipeline, along with their respective keys for the `--dataset` parameter:

| Dataset Name                   | `--dataset` Key       | # Bands | Wavelengths | # Images | Expected Extension | File Loader Detail / Required Accompanying Files                           |
| :----------------------------- | :---------------------- | :-----: | :----------: | :------: | :----------------: | :------------------------------------------------------------------------- |
| **50 Outdoor**           | `fiftyoutdoor`        |   33   | 400–720 nm |    50    |      `.mat`      | Custom Mat loader                                                          |
| **Agricultural Plant**   | `aphid`               |   237   | 436–965 nm |   361   |      `.npy`      | NumPy array loader                                                         |
| **ARAD1K16**             | `arad_1k_16`          |   16   | 400–1000 nm |   950   |      `.mat`      | HDF5 H5PY loader                                                           |
| **ARAD1K31**             | `arad_1k_31`          |   31   | 400–700 nm |   949   |      `.mat`      | HDF5 H5PY loader                                                           |
| **CAVE**                 | `cave`                |   31   | 400–700 nm |    32    |      `.mat`      | Mat loader                                                                 |
| **DeepHS-NIR**           | `deephsnir`           |   252   | 950–1700 nm |   718   |      `.bin`      | ENVI binary (requires corresponding`.hdr` file in the same directory)    |
| **DeepHS-VIS**           | `deephsvis`           |   224   | 400–1000 nm |   3405   |      `.bin`      | ENVI binary (requires corresponding`.hdr` file in the same directory)    |
| **DeepHS-VISCOR**        | `deephsviscor`        |   249   | 400–1000 nm |   1566   |      `.bin`      | ENVI binary (requires corresponding`.hdr` file in the same directory)    |
| **Harvard**              | `harvard`             |   31   | 420–720 nm |    77    |      `.mat`      | Mat loader                                                                 |
| **HOT-2024-NIR**         | `hotnir`              |   25   | 665–960 nm |   477   |      `.png`      | PNG frame loader (requires corresponding false-color`.jpg` file)         |
| **HOT-2024-RedNIR**      | `hotrednir`           |   15   | 600–850 nm |   348   |      `.png`      | PNG frame loader (requires corresponding false-color`.jpg` file)         |
| **HOT-2024-VIS**         | `hotvis`              |   16   | 470–600 nm |   1070   |      `.png`      | PNG frame loader (requires corresponding false-color`.jpg` file)         |
| **HSI Drive v2.0**       | `hsidrive20`          |   25   | 600–975 nm |   752   |      `.npy`      | NumPy loader (requires corresponding pseudocolor`.png` file)             |
| **HSI Road**             | `hsiroad`             |   25   | 600–960 nm |   380   |      `.tif`      | TIF image loader                                                           |
| **HSODBIT v2**           | `hsodbitv2`           |   200   | 400–1000 nm |   500   |      `.mat`      | HDF5 H5PY loader (requires corresponding color`.jpg` file)               |
| **HSSOD**                | `hs_sod`              |   81   | 380–720 nm |    60    |      `.h5`      | HDF5 H5PY loader (requires corresponding color`.jpg` file)               |
| **HyKo v2-NIR**          | `hykov2nir`           |   25   | 600–975 nm |    78    |      `.mat`      | Mat loader                                                                 |
| **HyKo v2-VIS**          | `hykov2vis`           |   16   | 470–630 nm |   163   |      `.mat`      | Mat loader                                                                 |
| **HyperBlood**           | `hyperblood`          |   128   | 377–1046 nm |    14    |      `.mat`      | Custom Mat loader                                                          |
| **HyperDrive-VNIR**      | `hyperdrivevnir`      |   24   | 660–900 nm |   504   |      `.npz`      | NumPy archive (must contain`cube.npy` file inside)                       |
| **HyperspectralCity v2** | `hyperspectralcityv2` |   128   | 450–950 nm |   1330   |      `.hsd`      | HSD raw data loader                                                        |
| **ICVL**                 | `icvl`                |   31   | 400–700 nm |   187   |      `.h5`      | HDF5 H5PY loader                                                           |
| **LIB-HSI**              | `libhsi`              |   204   | 400–1000 nm |   393   |      `.hdr`      | ENVI header (requires corresponding raw binary data`.raw`/`.dat` file) |
| **UM-EMM**               | `umemm`               |   33   | 400–720 nm |    3    |      `.mat`      | Mat loader                                                                 |
| **UM-LD 2015**           | `umld2015`            |   33   | 400–720 nm |    20    |      `.mat`      | Mat loader                                                                 |
| **UM-NS 2002**           | `umns2002`            |   31   | 410–710 nm |    8    |      `.mat`      | Mat loader                                                                 |
| **UM-NS 2004**           | `umns2004`            |   33   | 400–720 nm |    10    |      `.mat`      | Mat loader                                                                 |
| **UM-OS**                | `umos`                |   33   | 400–720 nm |    50    |      `.mat`      | Mat loader                                                                 |
| **UM-RI 2015**           | `umri2015`            |   33   | 400–720 nm |    33    |      `.mat`      | Mat loader                                                                 |
| **Virginia Tech Tree**   | `virginia_tech_tree`  |   420   | 400–1000 nm |    51    |      `.hdr`      | ENVI header (requires corresponding raw binary data file)                  |
| **Apple Fire Blight**    | `vnihdhiatlimafb`     |   204   | 400–1000 nm |   420   |      `.hdr`      | ENVI header (requires corresponding raw binary data file)                  |

---

## 📝 Citation

If you find our work or this project helpful, please consider citing our paper:

<!-- ```bibtex
@inproceedings{fu2026hypervision,
  title     = {{HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone}},
  author    = {Fu, Guanyiman and Li, Jingtao and Cheng, Zihang and Li, Zhuanfeng and Chen, Diqi and Xu, Yan and Liu, Xiangyu and Xiong, Fengchao and Lu, Jianfeng and Chen, Chengrong and Zhou, Jun},
  booktitle = {Proc. British Mach. Vis. Conf. (BMVC)},
  year      = {2026}
}
``` -->

```bibtex
@misc{fu2026hypervisionarxiv,
  title         = {{HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone}}, 
  author        = {Guanyiman Fu and Jingtao Li and Zihang Cheng and Zhuanfeng Li and Diqi Chen and Yan Xu and Xiangyu Liu and Fengchao Xiong and Jianfeng Lu and Chengrong Chen and Jun Zhou},
  year          = {2026},
  eprint        = {2605.17286},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.17286}
}
```

<!-- ---

## 📄 License

This project is licensed under the Apache 2.0 License. -->
