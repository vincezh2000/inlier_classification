import torch
import random
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from kitti import KITTIDataset
from easydict import EasyDict as edict
from utils import *
from transformer import FeatureExtractor
from trial_network import SiameseNetwork
import logging
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
    learning_rate = 0.0001 # try the new learning rate + double check the softmax results + input visualization Aditya + check dcp code (ex: batch norm layer) 
    epochs = 20
    CFG_DIR = r"/scratch/fsa4859/OverlapPredator/configs/test/kitti.yaml"
    config = edict(load_config(CFG_DIR))
    logging.warning("Starting the data loading process")
    dataset = KITTIDataset(config=config,split="test", data_augmentation=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'the length of the data loader is {len(loader)}')
    # define model,optimizer,loss
    #model=FeatureExtractor(emb_dims=256,n_blocks=5,dropout=0.1,ff_dims=512,n_heads=32)
    #model=(emb_dims=256,n_blocks=5,dropout=0.1,ff_dims=512,n_heads=32)
    #model.to(torch.device("cuda:0"))
    #optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    model=SiameseNetwork()
    model.to(torch.device("cuda:0"))
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    BCE_Loss=torch.nn.BCELoss()
   # mean_squared_loss=torch.nn.MSELoss()
    #triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # forward loop (training)
    losses=[]
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epochs):
            loss_total=0
            for index,data in enumerate(loader):
                if index>200:
                    break
                source=data[0].float().squeeze(0)
                target=data[1].float().squeeze(0)
                target=target.reshape((target.shape[0],target.shape[1],1))
                source=source.reshape((source.shape[0],source.shape[1],1))
                if source.shape[0]!=target.shape[0]:
                    if source.shape[0]>target.shape[0]:
                        source=source[0:target.shape[0],:,:]
                    else:
                        target=target[0:source.shape[0],:,:]
                source=source.to(device)
                target=target.to(device)
                #logging.warning("Loading data into model")
                src_out_net = model(source,target)
                src_out_net=torch.squeeze(src_out_net)
                print(f'printing unique elements in src_out net {src_out_net.unique()}')
                print(f'printing max element in src_out net {torch.max(src_out_net)}')
                print(f'printing min element in src_out net {torch.min(src_out_net)}')
                #for i in range(src_out_net.shape[0]):
                    #if src_out_net[i]>=0.5:
                        #src_out_net[i]=1
                #else:
                    #src_out_net[i]=0
            
                # read the indices of overlapping points
                path_to_overlapping_indices=r"/scratch/fsa4859/OverlapPredator/new_overlap_pairs/" +f"pair {index}" + ".npy"
                overlapping_indices=np.load(path_to_overlapping_indices,allow_pickle=True)
                overlapping_indices.astype(dtype=int,copy=False)
                overlapping_indices_torch=torch.from_numpy(overlapping_indices)
                overlapping_indices_torch=overlapping_indices_torch.to(device,dtype=torch.int64)
                print(f'printing overlapping indices torch {overlapping_indices_torch}')
                # initialize ground truth for both source and target
                gt_src=torch.zeros(source.shape[0],requires_grad=True).to(device)
                gt_tgt=torch.zeros(target.shape[0],requires_grad=True).to(device)
                # select the indices of the inliers from overlap torch and set the value to 1 in the target
                gt_src[overlapping_indices_torch[:,0]]=1
                gt_tgt[overlapping_indices_torch[:,1]]=1
                # compute the loss
                #loss_src=BCE_Loss(src_out_net,gt_src)
                print(f'ground truth source is {gt_src}')
                print(f'result from model is {src_out_net}')
                print(f'shape of source ground truth is {gt_src.shape}')
                loss_src=BCE_Loss(src_out_net,gt_src)
                # optimizer
                optimizer.zero_grad()
                loss_src.backward()
                optimizer.step()
                loss_total=loss_total+loss_src
            losses.append(loss_total/200)
            print(f"epoch {e}: loss:{loss_total/200}")
        
    torch.save(model,'/scratch/fsa4859/OverlapPredator/inlier_classification_model/model.pth')

    
    


    





    
     




