## 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks

[![arXiv](https://img.shields.io/badge/arXiv-2104.01797v2-00ff00.svg)](https://arxiv.org/pdf/2104.01797v2.pdf)

## Introduction

This repository contains the code and models for the following paper.

> [Monocular 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks](https://arxiv.org/pdf/2104.01797v2.pdf)  
> Cheng Yu, Bo Wang, Bo Yang, Robby T. Tan  
> Computer Vision and Pattern Recognition, CVPR 2021.


Overview of the proposed method:

<p align="center"><img src="3DMPP_framework.png" width="98%"></p>

## Installation

### Dependencies
[Pytorch](https://pytorch.org/) >= 1.5<br>
Python >= 3.6<br>

Create an enviroment. 
```
conda create -n 3dmpp python=3.6
conda activate 3dmpp
```
Install the latest version of pytorch (tested on pytorch 1.5 - 1.7) based on your OS and GPU driver installed following [install pytorch](https://pytorch.org/). For example, command to use on Linux with CUDA 11.0 is like:
```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

Install dependencies
```
pip install - r requirements.txt
```

Build the Fast Gaussian Map tool:

```
cd lib/fastgaus
python setup.py build_ext --inplace
cd ../..
```

## Models and Testing Data

### Pre-trained Models

Download the pre-trained model and processed human keypoint files [here](https://www.dropbox.com/s/n1twh0v5cyzd0z9/3DMPP.zip?dl=0), and unzip the downloaded zip file to this project's root directory, two folders are expected to see after doing that (i.e., `./ckpts` and `./mupots`).

### MuPoTS Dataset
MuPoTS eval set is needed to perform evaluation as the results reported in Table 3 in the main paper, which is available on the [MuPoTS dataset website](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). You need to download the `mupots-3d-eval.zip` file, unzip it, and run `get_mupots-3d.sh` to download the dataset. After the download is complete, a `MultiPersonTestSet.zip` is avaiable, ~5.6 GB. Unzip it and move the folder `MultiPersonTestSet` to the root directory of the project to perform evaluation on MuPoTS test set. Now you should see the following directory structure. 
```
${3D-Multi-Person-Pose_ROOT}
|-- ckpts              <-- the downloaded pre-trained Models
|-- lib
|-- MultiPersonTestSet <-- the newly added MuPoTS eval set
|-- mupots             <-- the downloaded processed human keypoint files
|-- util
|-- 3DMPP_framework.png
|-- calculate_mupots_btmup.py
|-- other python code, LICENSE, and README files
...
```

## Usage 

### MuPoTS dataset evaluation

#### 3D Multi-Person Pose Estimation Evaluation on MuPoTS Dataset

The following table is similar to Table 3 in the main paper, which shows the quantitative evaluations on MuPoTS-3D (best in **bold**). Please follow the instructions in the next section to reproduce the results (PCK and PCK_abs) in the table. 

| Group | Methods | PCK | PCK_abs |
|-------------|-------------------------------------------------------------|------------------|------------------|
| Person-centric (relative 3D pose) | [Mehta et al., 3DV'18](https://arxiv.org/pdf/1712.03453.pdf)| 65.0 | N/A |
| Person-centric (relative 3D pose) | [Rogez et al., IEEE TPAMI'19](https://arxiv.org/pdf/1803.00455.pdf) | 70.6 | N/A |
| Person-centric (relative 3D pose) | [Mehta et al., ACM TOG'20](https://dl.acm.org/doi/pdf/10.1145/3386569.3392410) | 70.4 | N/A |
| Person-centric (relative 3D pose) | [Cheng et al., ICCV'19](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.pdf) | 74.6 | N/A |
| Person-centric (relative 3D pose) | [Cheng et al., AAAI'20](https://ojs.aaai.org/index.php/AAAI/article/view/6689) | 80.5 | N/A |
| Camera-centric (absolute 3D pose) | [Moon et al., ICCV'19](https://openaccess.thecvf.com/content_ICCV_2019/papers/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.pdf) | 82.5 | 31.8 |
| Camera-centric (absolute 3D pose) | [Lin et al., ECCV'20](https://arxiv.org/pdf/2007.08943.pdf) | 83.7 | 35.2 |
| Camera-centric (absolute 3D pose) | [Zhen et al., ECCV'20](https://arxiv.org/pdf/2008.11469.pdf) | 80.5 | 38.7 |
| Camera-centric (absolute 3D pose) | [Li et al., ECCV'20](https://arxiv.org/pdf/2008.00206.pdf) | 82.0 | 43.8 |
| Camera-centric (absolute 3D pose) | [Cheng et al., AAAI'21](https://arxiv.org/pdf/2012.11806v3.pdf) | 87.5 | 45.7 |
| Camera-centric (absolute 3D pose) | [Our method](https://arxiv.org/pdf/2104.01797v2.pdf) | **89.6** | **48.0** |

#### Run evaluation on MuPoTS dataset with estimated 2D joints as input

We split the whole pipeline into several separate steps to make it more clear for the users. 

```
python calculate_mupots_topdown_pts.py
python calculate_mupots_topdown_depth.py
python calculate_mupots_btmup.py
python calculate_mupots_integrate.py
```
Please note that `python calculate_mupots_btmup.py` is going to take a while (~20 min depends on the mahine). 

To evaluate the person-centric 3D multi-person pose estimation:
```
python eval_mupots_pck.py
```
After running the above code, the following PCK (person-centric, pelvis-based origin) value is expected, which matches the number reported in Table 3, PCK 89.6 (percentage) in the paper.
```
...
Seq: 18
Seq: 19
Seq: 20
PCK_MEAN: 0.8994453169938017
```

To evaluate absolute PCK, run:
```
python eval_mupots_pck_abs.py
```

## Citation

If this work is useful for your research, please cite our paper. 
```
@InProceedings{Cheng_2021_CVPR,
    author    = {Cheng, Yu and Wang, Bo and Yang, Bo and Tan, Robby T.},
    title     = {Monocular 3D Multi-Person Pose Estimation by Integrating Top-Down and Bottom-Up Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7649-7659}
}
```
