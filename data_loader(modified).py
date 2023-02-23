import torch
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os
import h5py
from kitti import KITTIDataset
from easydict import EasyDict as edict
from utils import *
import random

if __name__== '__main__':
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    device=torch.device(dev)
    threshold_margin=0.1
    CFG_DIR = r"C:\Users\15512\Desktop\OverlapPredator\configs\test\kitti.yaml"
    print(type(CFG_DIR))
    print(CFG_DIR)
    config = edict(load_config(CFG_DIR))
    dataset = KITTIDataset(config=config,split="test", data_augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    pairs_dir=os.makedirs("pairs",exist_ok=True)
    for idx,data in enumerate(loader):
        source=data[0].float().squeeze(0)
        tgt=data[1].float().squeeze(0)
        rot=data[4].float().squeeze(0)
        trans=data[5].float().squeeze(0)
        source_transformed=((rot@source.T)+trans).T
        l2_dist = torch.cdist(source_transformed, tgt) # shape (source_shape,target_shape)
        threshold = torch.min(l2_dist)+threshold_margin
        overlap=torch.nonzero(l2_dist<threshold)
        torch.unique(overlap,dim=0)
        #gt_source=torch.zeros(source_transformed.shape[0],1)
        #gt_target=torch.zeros(tgt.shape[0],1)
        #gt_target[overlap[:,0],0]=1
        #gt_target[overlap[:,1],0]=1
        pair_info=os.path.join("pairs",f"pair {idx}.npy")
        #os.makedirs(os.path.join(pairs_dir,f"{idx}"))
        np.save(pair_info,overlap.numpy())
        
    #torch.unique(overlap,dim=0) # removes any redundant rows. 
    #print(f'overlap shape is{overlap.shape}')
    #print(overlap)

    # now i have the overlapping indices of the source and target point cloud. 
    # the positives will be the inlier indices of the target point cloud
    # the negatives will be the outlier indices of the target point cloud
    # the anchor will be the inlier indices of the source point cloud
    # by maximizing the distance between the anchor and negatives while 
    # minimizing the distance between the anchor and positives. 
    # contrastive loss enforces the distance to be zero.
    # 

    # start forming anchor, positives and negatives. 
    #anchors=torch.zeros(overlap.shape[0],3)
    #positives=torch.zeros(overlap.shape[0],3)
    #all_indices=[i for i in range(tgt.shape[0])]
    #positive_indices=[]
    

    #for j in range(anchors.shape[0]):
        #anchors[j,:]=source[overlap[j][0],:]
        #positives[j,:]=tgt[overlap[j][1],:]
        #positive_indices.append(int(overlap[j][1]))

    #neagtive_indices_all=[i for i in all_indices if i not in positive_indices]
    #negative_samples = random.choices(neagtive_indices_all,k=overlap.shape[0])
    #negatives=tgt[negative_samples,:]

    #print(f'the shape of negatives is {negatives.shape}')

    #print(f'shape of anchors is {anchors.shape}')
    #print(f'shape of positives is {positives.shape}')
    #print(f'the length of all indices is {len(all_indices)}')
    #print(f'the length of positive indices is {len(positive_indices)}')
    #print(anchors)
    #print(positives)
    #print(negatives)

    # try the triplet loss
    #triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    #output = triplet_loss(anchors, positives, negatives)
    #print(output.shape)
    #print(output)

    

    
  
        