import torch
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os
import h5py
from kitti import KITTIDataset
from easydict import EasyDict as edict
from utils import *


if __name__== '__main__':
    CFG_DIR = r"C:\Users\15512\Desktop\OverlapPredator\configs\test\kitti.yaml"
    print(type(CFG_DIR))
    print(CFG_DIR)
    config = edict(load_config(CFG_DIR))
    dataset = KITTIDataset(config=config,split="test", data_augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    os.makedirs("labels",exist_ok=True)
    for idx,data in enumerate(loader):
        source=data[0].float().squeeze(0)
        tgt=data[1].float().squeeze(0)
        rot=data[4].float().squeeze(0)
        trans=data[5].float().squeeze(0)
        source_transformed=((rot@source.T)+trans).T
        l2_dist = torch.cdist(source_transformed, tgt)
        threshold = 15 * torch.min(l2_dist)
        overlap_indexes = torch.nonzero(l2_dist < threshold)
        gt = torch.zeros(source_transformed.shape[0], 2)
        gt[overlap_indexes[:, 0], 0] = 1
        gt[overlap_indexes[:, 0], 1] = 1
        GT_PATH = os.path.join("labels", f"{idx}.npy")
        np.save(GT_PATH,gt.numpy())

           
    def validate_grount_truth(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


    
    
   
 
    
   



    
        
        
      

    
        