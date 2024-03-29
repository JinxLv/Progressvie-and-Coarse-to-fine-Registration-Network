# Progressvie-and-Coarse-to-fine-Registration-Network
The implementation of our paper "Joint Progressive and Coarse-to-fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion". 

The motivation of this work is to decompose the deformation field in both progressive and coarse-to-fine manner for alleviating the difficulty of prediction. Specifically, we first built a unified CNN which can decompose the deformation filed in a coarse-to-fine manner, and then proposed the DFI and NFF modules for the progressive decomposition relying on light-weight decoding blocks instead of heavy-weight CNN models, i.e. VTN. 

For more details, please refer to our [paper](https://ieeexplore.ieee.org/document/9765391).

<img src="./Figure/framework.jpg" width="700px">

## Install
The packages and their corresponding version we used in this repository are listed in below.

- Tensorflow==1.15.4
- Keras==2.3.1
- tflearn==0.5.0

## Training
After configuring the environment, please use this command to train the model.

```sh
python train.py -g 0 --batch 1 -d datasets/brain.json -b PCNet -n 1 --round 10000 --epoch 10
```

## Testing
Use this command to obtain the testing results.
```sh
python predict.py -g 0 --batch 1 -d datasets/brain.json -c weights/Apr06-1516
```

## Pre-trained model and testing data on LPBA40
The [pre-trained model](https://drive.google.com/file/d/1NndVW8beu-fYjDP2mVsOf-WRnX2NCZUQ/view?usp=sharing) and [testing data](https://drive.google.com/file/d/1tU42wwc1qLlwJEI3IHcP30XqOtW0j7hb/view?usp=sharing) are available. Please unzip these files, and move the `lpba_val.h5` to `/datasets/` folder.

## Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@ARTICLE{9765391,
  author={Lv, Jinxin and Wang, Zhiwei and Shi, Hongkuan and Zhang, Haobo and Wang, Sheng and Wang, Yilang and Li, Qiang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Joint Progressive and Coarse-to-Fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion}, 
  year={2022},
  volume={41},
  number={10},
  pages={2788-2802},
  doi={10.1109/TMI.2022.3170879}}
```

## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.


