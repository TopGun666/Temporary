# Temporary
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-big-to-small-multi-scale-local-planar/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=from-big-to-small-multi-scale-local-planar) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-big-to-small-multi-scale-local-planar/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=from-big-to-small-multi-scale-local-planar)

## Preamble
This codebase was developed and tested with python 3.6.5, Pytorch 1.5.0, and CUDA 10.0 on Ubuntu 18.04. It is based on [Han Lee's BTS implementation](https://github.com/cogaplex-bts/bts.git).

## Prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set
```shell
$ cd ~/paper/Temporary/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```
## Prepare [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip) official ground truth depth maps
Download the ground truth depthmaps from this link [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).\
Then,
```
$ cd ~/paper/dataset
$ mkdir kitti_dataset && cd kitti_dataset
$ mv ~/Downloads/data_depth_annotated.zip .
$ unzip data_depth_annotated.zip
```

## PyTorch Implementation
[[./pytorch/]](./pytorch/)

## Model Zoo
### KITTI Eigen Split

| Base Network |  cap  |   d1  |   d2  |   d3  | AbsRel | SqRel |  RMSE | RMSElog | SILog | log10 | #Params |          Model Download          |
|:------------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-------:|:-----:|:-----:|:-------:|:--------------------------------:|
| ResNet50     | 0-80m | 0.954 | 0.992 | 0.998 |  0.061 | 0.250 | 2.803 |   0.098 | 9.030 | 0.027 |   49.5M | [bts_eigen_v2_pytorch_resnet50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnet50.zip)  |
| ResNet101    | 0-80m | 0.954 | 0.992 | 0.998 |  0.061 | 0.261 | 2.834 |   0.099 | 9.075 | 0.027 |   68.5M | [bts_eigen_v2_pytorch_resnet101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnet101.zip) |
| ResNext50    | 0-80m | 0.954 | 0.993 | 0.998 |  0.061 | 0.245 | 2.774 |   0.098 | 9.014 | 0.027 |   49.0M | [bts_eigen_v2_pytorch_resnext50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnext50.zip)  |
| ResNext101   | 0-80m | 0.956 | 0.993 | 0.998 |  0.059 | 0.241 | 2.756 |   0.096 | 8.781 | 0.026 |  112.8M | [bts_eigen_v2_pytorch_resnext101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnext101.zip)  |
| DenseNet121  | 0-80m | 0.951 | 0.993 | 0.998 |  0.063 | 0.256 | 2.850 |   0.100 | 9.221 | 0.028 |   21.2M | [bts_eigen_v2_pytorch_densenet121](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet121.zip) |
| DenseNet161  | 0-80m | 0.955 | 0.993 | 0.998 |  0.060 | 0.249 | 2.798 |   0.096 | 8.933 | 0.027 |   47.0M | [bts_eigen_v2_pytorch_densenet161](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet161.zip) |

### NYU Depth V2

| Base Network |   d1  |   d2  |   d3  | AbsRel | SqRel |  RMSE | RMSElog |  SILog | log10 | #Params |         Model Download         |
|:------------:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-------:|:------:|:-----:|:-------:|:------------------------------:|
| ResNet50     | 0.865 | 0.975 | 0.993 |  0.119 | 0.075 | 0.419 |   0.152 | 12.368 | 0.051 |   49.5M | [bts_nyu_v2_pytorch_resnet50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_resnet50.zip) |
| ResNet101    | 0.871 | 0.977 | 0.995 |  0.113 | 0.068 | 0.407 |   0.148 | 11.886 | 0.049 |   68.5M | [bts_nyu_v2_pytorch_resnet101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_resnet101.zip) |
| ResNext50    | 0.867 | 0.977 | 0.995 |  0.116 | 0.070 | 0.414 |   0.150 | 12.186 | 0.050 |   49.0M | [bts_nyu_v2_pytorch_resnext50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_resnext50.zip)  |
| ResNext101   | 0.880 | 0.977 | 0.994 |  0.111 | 0.069 | 0.399 |   0.145 | 11.680 | 0.048 |  112.8M | [bts_nyu_v2_pytorch_resnext101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_resnext101.zip)  |
| DenseNet121  | 0.871 | 0.977 | 0.993 |  0.118 | 0.072 | 0.410 |   0.149 | 12.028 | 0.050 |   21.2M | [bts_nyu_v2_pytorch_densenet121](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet121.zip) |
| DenseNet161  | 0.885 | 0.978 | 0.994 |  0.110 | 0.066 | 0.392 |   0.142 | 11.533 | 0.047 |   47.0M | [bts_nyu_v2_pytorch_densenet161](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip) |

Note: Modify arguments '--encoder', '--model_name', '--checkpoint_path' and '--pred_path' accordingly.

## Live Demo
Finally, we attach live 3d demo implementations for Pytorch. \
For best performance, get correct intrinsic values for your webcam and put them in bts_live_3d.py. \
Sample usage for PyTorch:
```
$ cd ~/paper/Temporary/pytorch
$ python bts_live_3d.py --model_name bts_nyu_v2_pytorch_densenet161 \
--encoder densenet161_bts \
--checkpoint_path ./models/bts_nyu_v2_pytorch_densenet161/model \
--max_depth 10 \
--input_height 480 \
--input_width 640
```

## License
This Software is licensed under GPL-3.0-or-later.