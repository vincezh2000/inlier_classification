## Inlier Classification of Low Overlap Point Clouds

### [Inlier Classification of Low Overlapping Point Clouds]

### Contact
If you have any questions, please let us know: 
- Fady Algyar {fsa4859@nyu.edu}

## Ground Truth Data Loading

- To train our network, we need to prepare the dataset so that each point has a binary classification of whether it is an inlier or not. 
- The model is being trained on KITTI dataset loaded with Predator dataloader (low overlapping point clouds).
- Based on the Euclidean Distance between the points in the point clouds, we can set some threshold value based on which the classification of inliers is performed. 
- The "overlap_pairs" is the folder containing the ground truth data.
- The /overlap_pairs contains numpy arrays. 
- Each numpy array contain the indices of the inliers for a given pair of point clouds (source and target).

![image](https://github.com/fsa4859/inlier_classification/assets/69100847/31d44a19-7d42-4479-ba10-18f5b8f0e336)

![image](https://github.com/fsa4859/inlier_classification/assets/69100847/15cd12bb-8732-46ad-bd25-e3e8ae79a046)


**Note**:

### Model Training and Evaluation

- To train the model run main_trial_final.py
- To perform model evaluation, run model_inference.py

### Datasets
For KITTI dataset, please follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set

### Acknowledgments

- To train the model run main_trial_final.py
- To perform model evaluation, run model_inference.py



### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/overlappredator/OverlapPredator.git
virtualenv predator; source predator/bin/activate
cd OverlapPredator; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
in your working folder.

### Datasets
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
### Acknowledgments

 We thank the respective authors for open sourcing their methods. We would also like to thank reviewers, especially reviewer 2 for his/her valuable inputs. 
