import torch
torch.cuda.empty_cache()
import random
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from kitti import KITTIDataset
from easydict import EasyDict as edict
from utils import *
from classification_model import Net
from visualize import plot_loss
from transformer import FeatureExtractor
#load dataset
# what is the input? the input should be the source_transformed and taregt point cloud (shape_source,3),(shape_target,3)
# what is the ground truth? the ground truth is what comes out from the dataset (shape_point_cloud,2) 
#  how can we define loss function then? the network should output tensors of the inliers indices (shape_point_cloud,2) and
# loss is the difference between the predicted inliers and provided ground truth.
# the input is just the input ground truth without any transformations.
# load the data in the traditional method
if __name__=='__main__':
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    device=torch.device(dev)
    learning_rate = 0.01
    epochs = 10
    CFG_DIR = r"C:\Users\15512\Desktop\OverlapPredator\configs\test\kitti.yaml"
    config = edict(load_config(CFG_DIR))
    dataset = KITTIDataset(config=config,split="test", data_augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # define model,optimizer,loss
    model=FeatureExtractor(emb_dims=128,n_blocks=5,dropout=0.1,ff_dims=512,n_heads=32)
    model.to(torch.device("cuda:0"))
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # forward loop (training)
    losses=[]
    #losses=losses.to(torch.device("cuda:0"))
    for j in range(epochs):
        loss_total=0
        for index,data in enumerate(loader):
            if index>100:
                break
            source=data[0].float().squeeze(0)
            target=data[1].float().squeeze(0)
            target=target.reshape((target.shape[0],target.shape[1],1))
            source=source.reshape((source.shape[0],source.shape[1],1))
            print(f'shape of new source is {source.shape}')
            print(f'shape of new target is {target.shape}')
            if source.shape[0]!=target.shape[0]:
                if source.shape[0]>target.shape[0]:
                    source=source[0:target.shape[0],:,:]
                else:
                    target=target[0:source.shape[0],:,:]
            print("asserting shapes are equal")
            print(f'shape of new source after equalling is {source.shape}')
            print(f'shape of new target after equalling is {target.shape}')
            source=source.to(device)
            target=target.to(device)
            src_emb,tgt_emb = model(source,target)
            # read the indices of overlapping points
            path_to_overlapping_indices=r"C:\Users\15512\Desktop\OverlapPredator\overlap_pairs" +f"pair {index}" + ".npy"
            overlapping_indices=np.load(path_to_overlapping_indices,allow_pickle=True)
            overlapping_indices_torch=torch.from_numpy(overlapping_indices)
            overlapping_indices_torch=overlapping_indices_torch.to(device)
            # initialize anchors,positives 
            anchors=torch.zero(overlapping_indices_torch.shape[0],512).to(device) # 512 is the embedding dimension
            positives=torch.zero(overlapping_indices_torch.shape[0],512).to(device) # 512 is the embedding dimension
            all_indices=[i for i in range(overlapping_indices_torch.shape[0])]
            positive_indices=[]
            # select the indices of the overlapping points from the model output
            # this will form anchors and positives
            for j in range(overlapping_indices_torch.shape[0]):
                anchors[j,:]=src_emb[overlapping_indices_torch[j][0],:,:]
                positives[j,:]=tgt_emb[overlapping_indices_torch[j][1],:,:]
                positive_indices.append(int(overlapping_indices_torch[j][1]))
            neagtive_indices_all=[i for i in all_indices if i not in positive_indices]
            negative_samples = random.choices(neagtive_indices_all,k=overlapping_indices_torch.shape[0])
            negatives=tgt_emb[negative_samples,:,:].to(device)
            # compute the loss function
            #loss=loss_fn(output,gt_torch)
            loss=triplet_loss(anchors, positives, negatives)
            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total=loss_total+loss
        losses.append(loss_total/101)
        print(f"epoch {j}: loss:{loss}")
    
    


    





    
     




