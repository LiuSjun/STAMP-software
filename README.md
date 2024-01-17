# STAMP Software

> **[Beijing Normal University, CHEN-Lab](http://www.chen-lab.club/)**
> 
> Contributors: [Shuaijun Liu](https://alex1234.github.io/), Dong Qi, Xuehong Chen, Xiuchun Dong, Ping Huang, Peng Yang, Jin Chen
> 
> Resources: [[`Academic Paper`]] [[`Demo`]]

<p align="center">
  <img src="pic/Flowchart_Stamp.png?raw=true" width="50.25%" />
</p>

## Overview
**STAMP (Segment Anything Model for Planted Fields)** is an adaptive model designed for segmentation of planted fields from remote sensing imagery. Building upon the 'Segment Anything Model', it boasts enhanced zero-shot performance in remote sensing image analysis.

<p align="center">
  <img src="pic/Fig2.png?raw=true" width="37.25%" />
</p>

## Installation and Requirements

### System Requirements
- Python 3.8+
- PyTorch 1.7.0+
- CUDA 11.0+ (Recommended)

### Installation Instructions
STAMP can be easily installed via pip or by cloning the repository.

### Additional Dependencies
For mask post-processing and running example notebooks, additional packages are required.

### Prerequisites
- numpy 1.24.3
- torchvision 0.8+
- GDAL, OpenCV
- [Albumentations](https://pypi.org/project/albumentations/) 1.3.1+

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install STAMP:

```
pip install STAMP.git
```

or clone the repository locally and install with

```
git clone git@github.com:STAMPg.git
cd STAMP; pip install -e .
```

The following optional dependencies are necessary for mask post-processing,`jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Getting Started with STAMP

First download STAMP. Then the model can be used in just a few lines to get masks:

```
from STAMP import auotSTAMP
stamp = auotSTAMP["<model_type>"]
predictor = stamp(pic)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from STAMP import STAMPWindow
stampWindow = STAMPWindow()
mask_generator = stampWindow(your_image)
masks = mask_generator.generate(your_image)
```

For detailed examples, see our [notebooks](/notebooks/STMAP_example.ipynb).

<p float="left">
  <img src="pic/Fig3.png?raw=true" width="36.1%" />
  <img src="pic/Fig4.png?raw=true" width="48.9%" />
</p>

## Demonstrations

### Software Demo
Explore the `STAMP` one-page app for intuitive mask prediction. Detailed instructions are available in [`STAMPWindow.md`](https://github.com/LiuSjun/STAMP/README.md).

#### Demo Steps
1. **Start the Demo**: Double-click 'STAMP.exe'.
   <p align="center">
     <img src="pic/Step1.gif?raw=true" width="50.25%" />
   </p>

2. **Select and Open Image**.
   <p align="center">
     <img src="pic/step2.gif?raw=true" width="50.25%" />
   </p>

3. **Import or Auto-Select Processing Area**.
   <p align="center">
     <img src="pic/step3.gif?raw=true" width="50.25%" />
   </p>

4. **Extract Missing PFs** (manually or automatically).
   <p align="center">
     <img src="pic/step4.gif?raw=true" width="50.25%" />
     <img src="pic/step5.gif?raw=true" width="50.25%" />
   </p>

### FieldSeg-DA Integration
Combining STAMP with FieldSeg-DA for enhanced accuracy:

<p align="center">
  <img src="pic/FieldSegDA.png?raw=true" width="50.25%" />
</p>

## Model Selection
STAMP offers three model versions to cater to different time constraints:

<p align="center">
  <img src="pic/Model_Select.gif?raw=true" width="50.25%" />
</p>

## License and Citation

### License
STAMP is licensed under [beta 3.0.2](LICENSE).

### How to Cite
If you use STAMP or FieldSeg-DA in your research, please use the following BibTeX entry.

```
@article{kirillov2023stamp,
  title={STAMP},
  author={Liu Shuaijun, Dong Qi, Dong Chunxiu, Huang Ping, Yang Peng, Chen Xuehong, Chen Jin},
  journal={arXiv:####},
  year={2023}
}
@article{liu2022deep,
  title={A deep learning method for individual arable field (IAF) extraction with cross-domain adversarial capability},
  author={Liu, Shuaijun and Liu, Licong and Xu, Fei and Chen, Jin and Yuan, Yuhen and Chen, Xuehong},
  journal={Computers and Electronics in Agriculture},
  volume={203},
  pages={107473},
  year={2022},
  publisher={Elsevier}
}
