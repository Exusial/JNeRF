<!-- # JNeRF -->
<div align="center">
<img src="docs/logo.png" height="200"/>
</div>

## Introduction
JNeRF is an NeRF benchmark based on [Jittor](https://github.com/Jittor/jittor). JNeRF supports Instant-NGP capable of training NeRF in 5 seconds and achieves similar performance and speed to the paper.

5s training demo of Instant-NGP implemented by JNeRF:

<img src="docs/demo_5s.gif" width="300"/>

## Install
JNeRF environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
sudo apt-get install tcl-dev tk-dev python3-tk
git clone https://github.com/Jittor/JNeRF
cd JNeRF
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JNeRF**
 
You can add ```export PYTHONPATH=$PYTHONPATH:{your_path_to_jnerf}/JNeRF/python``` into ```~/.bashrc```, and run
```shell
source ~/.bashrc
```

## Getting Started

### Datasets

We use fox datasets and blender lego datasets for training demonstrations. 

#### Fox Dataset
We provided fox dataset in this repository at `./data/fox`.

#### Lego Dataset
You can download the lego dataset in nerf_example_data.zip at https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1. And move `lego` folder to `./data/lego`.

#### Customized Datasets

If you want to train JNerf with your own dataset, then you should follow the format of our datasets. You should split your datasets into training, validation and testing sets. Each set should be paired with a json file that describes the camera parameters of each images.

### Config

We organize our configs of JNeRF in projects/. You are referred to `./projects/ngp/configs/ngp_base.py` to learn how it works.

### Train & Test

Train and test on `lego` scene are combined in a single command. It should be noted that since jittor is a just-in-time compilation framework, it will take some time to compile on the first run.
```shell
python tools/run_net.py --config-file ./projects/ngp/configs/ngp_base.py
```
### GPU Supporting

JNeRF must run on GPU with sm arch no less than sm_61 (GTX 10x0 and above). If you want to enable FullyFusedMLP, then the sm arch of the GPU must be no less than sm_75(RTX 20x0 and above). JNeRF will use original MLPs if the requirements are not meet.

## Performance

Instant-ngp implemented by JNeRF achieves similar performance and speed to the paper. The performance comparison can be seen in the table below and training speed of JNeRF-NGP on RTX 3090 is about 133 iters/s. 


|    Models     |    implementation      | Dataset | PSNR |
|----|---|---|---|
| Instant-ngp | paper | lego | 36.39(5min) |
| Instant-ngp | JNeRF | lego | 36.41(5min) |
| NeRF        | JNeRF | lego | 32.49 |

## Plan of Models

JNeRF will support more valuable NeRF models in the future, if you are also interested in JNeRF and want to improve it, welcome to submit PR!

<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>

- :heavy_check_mark: Instant-NGP
- :heavy_check_mark: NeRF
- :clock3: Mip-NeRF
- :heavy_plus_sign: Plenoxels
- :heavy_plus_sign: StylizedNeRF
- :heavy_plus_sign: NeRF-Editing
- :heavy_plus_sign: DrawingInStyles
- :heavy_plus_sign: NSVF
- :heavy_plus_sign: NeRFactor
- :heavy_plus_sign: pixelNeRF
- :heavy_plus_sign: Recursive-NeRF
- :heavy_plus_sign: StyleNeRF
- :heavy_plus_sign: EG3D
- :heavy_plus_sign: ...

## Contact Us

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083

<img src="docs/qrcode.jpg" width="200"/>


## Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
