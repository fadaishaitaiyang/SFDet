# SFDet



## Dependencies

Python == 3.7.16

PyTorch == 1.7.0

MMDetection == 2.20.0

MMCV == 1.4.6

Numpy == 1.21.2

## Installation

The basic installation follows with [[mmdetection]](https://github.com/open-mmlab/mmdetection) [[document]](https://mmdetection.readthedocs.io/en/latest/). It is recommended to use manual installation.

## Dataset

We use dataset UTDAC2020, the download link of which is shown as follows.

https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing

After downloading all datasets, create UTDAC2020 document.

```
$ cd data
$ mkdir UTDAC2020
```

It is recommended to symlink the dataset root to `$data`.

```

├── data
│   ├── UTDAC2020
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
```



Other underwater dataset: https://github.com/mousecpn/Collection-of-Underwater-Object-Detection-Dataset

## Train

If you want to use Pascal VOC or COCO dataset, lease change the dataset type under the `roitransformer_r50_fpn_1x_coco.py` file.

```
$ python tools/train.py 
```

## Test

```
$ python tools/test.py configs/sfdet/sfdet_1x_utdac.py <path/to/checkpoints>
```


```

## Acknowledgement

This work is suported by Science and Technology Development Fund of Macau (0008/2019/A1, 0010/2019/AFJ, 0025/2019/AKP).

And thanks MMDetection team for the wonderful open source project!
