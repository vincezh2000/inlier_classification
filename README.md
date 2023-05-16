## Inlier Classification of Low Overlap Point Clouds

### [Inlier Classification of Low Overlapping Point Clouds]

### Contact
If you have any questions, please let us know: 
- Fady Algyar {fsa4859@nyu.edu}

## Instructions
This code has been tested on 
- Python 3.8.5, PyTorch 1.7.1, CUDA 11.2, gcc 9.3.0, GeForce RTX 3090/GeForce GTX 1080Ti

**Note**: We observe random data loader crashes due to memory issues, if you observe similar issues, please consider reducing the number of workers or increasing CPU RAM. We now released a sparse convolution-based Predator, have a look [here](https://github.com/ShengyuH/OverlapPredator.Mink.git)!

### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/overlappredator/OverlapPredator.git
virtualenv predator; source predator/bin/activate
cd OverlapPredator; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
in your working folder.

### Datasets and pretrained models
For KITTI dataset, please follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set
### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [FCGF](https://github.com/chrischoy/FCGF) (KITTI preprocessing)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch) (KPConv backbone)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet) (3DMatch preparation)
- [MultiviewReg](https://github.com/zgojcic/3D_multiview_reg) (3DMatch benchmark)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (Transformer part)
- [DGCNN](https://github.com/WangYueFt/dgcnn) (self-gnn)
- [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet preprocessing and evaluation)

 We thank the respective authors for open sourcing their methods. We would also like to thank reviewers, especially reviewer 2 for his/her valuable inputs. 
