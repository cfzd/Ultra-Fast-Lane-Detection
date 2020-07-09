# Ultra-Fast-Lane-Detection
The implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

Updates: Our paper has been accepted by ECCV2020.

![alt text](vis.jpg "vis")

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).

# Demo 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=lnFbAG3GBN4
" target="_blank"><img src="http://img.youtube.com/vi/lnFbAG3GBN4/0.jpg" 
alt="Demo" width="240" height="180" border="10" /></a>


# Install

a. Clone the project

```
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
cd Ultra-Fast-Lane-Detection
```

b. Create a conda virtual environment and activate it

```
conda create -n lane-det python=3.7 -y
conda activate lane-det
```

c. Install dependencies

```
# If you don't have pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

pip install -r requirements.txt
```

d. Install CULane evaluation tools. This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***

```
# First you need to install OpenCV C++. 
# After installation, make a soft link of OpenCV include path.
ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

cd evaluation/culane
make
```

e. Data preparation

Download [CULane](https://xingangpan.github.io/projects/CULane.html) and [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$CULANEROOT` and `$TUSIMPLEROOT`. For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```
cd $TUSIMPLEROOT
python scripts/convert_tusimple.py --root $TUSIMPLEROOT
# this will generate segmentations and 2 list files train_gt.txt test.txt
```

# Get Started
Copy a config from ```configs/culane.py``` or ```configs/tusimple.py```, then
modifiy ```data_root```, ```log_path``` and other settings in your config.

***

For single gpu training, run
```
python train.py configs/path_to_your_config
```
For multi-gpu training, run
```
sh launch_training.sh

# If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.
```
or
```
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```

***

Besides config style settings, we also support command line style one. You can override a setting like
```
python train.py configs/path_to_your_config --batch_size 8
```
The ```batch_size``` will be set to 8 during training.

***

To visualize the log with tensorboard, run

```
tensorboard --logdir log_path --bind_all
```

# Trained models
We provide two trained Res-18 models on CULane and Tusimple.

|  Dataset | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti |    Model    |
|:--------:|:------------:|:----------------:|:-------------------:|:-----------:|
| Tusimple |     95.87    |       95.82      |         306         | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag)|
|  CULane  |     68.4     |       69.7       |         324         | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |

For evaluation, run
```
mkdir tmp

python test.py configs/culane.py --test_model path_to_culane_18.pth --test_work_dir ./tmp

python test.py configs/tusimple.py --test_model path_to_tusimple_18.pth --test_work_dir ./tmp
```

Same as training, multi-gpu evaluation is also supported.

# Citation

```
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}
```