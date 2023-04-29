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
from loss_chamfer import *
import logging
import sklearn
import gc
from sklearn.neighbors import KDTree


def chamfer_distance_sklearn(array1,array2):
    array1=array1.cpu()
    array2=array2.cpu()
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
        dist_torch=torch.tensor(dist)
        return dist_torch

'''
if __name__=='__main__':
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    device=torch.device(dev)
    gc.collect()
    torch.cuda.empty_cache()
    #learning_rate = 0.00001 # try the new learning rate + double check the softmax results + input visualization Aditya + check dcp code (ex: batch norm layer) 
    #epochs = 20
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
    #model=SiameseNetwork()
    #model.to(torch.device("cuda:0"))
    #optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    #BCE_Loss=torch.nn.BCELoss()
    chamfer_loss=ChamferLoss() # New: Added chamfer loss 
   # mean_squared_loss=torch.nn.MSELoss()
    #triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # forward loop (training)
    #losses=[]
    with torch.autograd.set_detect_anomaly(True):
        for index,data in enumerate(loader):
            if index>0:
                break
            source=data[0].float().squeeze(0)
            target=data[1].float().squeeze(0)
            rot=data[4].float().squeeze(0)
            print(f'rotation is {rot} and shape is {rot.shape}')
            trans=data[5].float().squeeze(0)
            print(f'translation is {trans} and shape is {trans.shape}')
            target=target.reshape((target.shape[0],target.shape[1],1))
            source=source.reshape((source.shape[0],source.shape[1],1))
            if source.shape[0]!=target.shape[0]:
                 if source.shape[0]>target.shape[0]:
                     source=source[0:target.shape[0],:,:]
                 else:
                     target=target[0:source.shape[0],:,:]
            translation_shifted=torch.tensor([1,1,1])
            translation_shifted=torch.reshape(translation_shifted,(1,3))
            number_points_source=source.shape[0]
            print(f'number of points in the source is {number_points_source} and type is {type(number_points_source)}')
            print(f'shape of translation shifted is {translation_shifted.shape}')
            translation_shifted=translation_shifted.repeat(number_points_source,1)
            translation_shifted=torch.reshape(translation_shifted,(translation_shifted.shape[0],translation_shifted.shape[1],1))
            print(f'shape of translation shifted is {translation_shifted.shape}')
            print(f'shape of source is {source.shape}')
            source_shifted=source+translation_shifted
            print(f'shape of source shifted is {source_shifted} and shape is {source_shifted.shape}')
            loss_chamfer_same=chamfer_loss(source,source_shifted)
            loss_chamfer_srctgt=chamfer_loss(source,target)
            print(f'chamfer loss for same  is {loss_chamfer_same}')
            print(f'chamfer loss for different is {loss_chamfer_srctgt}')
            ######### testing sklearn ###########
            ######### the shape of source and target should be as follows (1,no_points,3) #######
            source_sklearn=torch.reshape(source,(source.shape[2],source.shape[0],source.shape[1]))
            target_sklearn=torch.reshape(target,(target.shape[2],target.shape[0],target.shape[1]))
            dis=chamfer_distance_sklearn(source_sklearn,target_sklearn)
            print(f'chamfer distance from sklearn is {dis} ')
            '''