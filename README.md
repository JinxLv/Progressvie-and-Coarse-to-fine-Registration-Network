# Progressvie-and-Coarse-to-fine-Registration-Network
The implementation of [Joint Progressive and Coarse-to-fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion](https://arxiv.org/abs/2109.12384)

## Install
The packages and their corresponding version we used in this repository are listed in below.

- Tensorflow==1.15.4
- Keras==2.3.1
- tflearn==0.5.0

## Training
After configuring the environment, please use this command to train the model.

```sh
python train.py -g 0 --batch 1 -d datasets/brain.json -b PCNET -n 1 --round 10000 --epoch 10
```

## Testing
Use this command to obtain the testing results.
```sh
python predict.py -g 0 --batch 1 -d datasets/brain.json -c weights/Dec09-1849
```


### Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@article{lv2021joint,
  title={Joint Progressive and Coarse-to-fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion},
  author={Lv, Jinxin and Wang, Zhiwei and Shi, Hongkuan and Zhang, Haobo and Wang, Sheng and Wang, Yilang and Li, Qiang},
  journal={arXiv preprint arXiv:2109.12384},
  year={2021}
}

## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.

}
```
