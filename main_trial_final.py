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
from misalignment import *
import logging
import gc
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
    gc.collect()
    torch.cuda.empty_cache()
    learning_rate = 0.000001 # try the new learning rate + double check the softmax results + input visualization Aditya + check dcp code (ex: batch norm layer) 
    epochs = 30
    CFG_DIR = r"/scratch/fsa4859/OverlapPredator/configs/test/kitti.yaml"
    config = edict(load_config(CFG_DIR))
    logging.warning("Starting the data loading process")
    dataset = KITTIDataset(config=config,split="test", data_augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'the length of the data loader is {len(loader)}')
    # define model,optimizer,loss
    #model=FeatureExtractor(emb_dims=256,n_blocks=5,dropout=0.1,ff_dims=512,n_heads=32)
    #model=(emb_dims=256,n_blocks=5,dropout=0.1,ff_dims=512,n_heads=32)
    #model.to(torch.device("cuda:0"))
    #optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    model=SiameseNetwork()
    model.to(torch.device("cuda:0"))
    logging.warning("Done transferring model to cuda device")
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    BCE_Loss=torch.nn.BCELoss()
    logging.warning("Done defining loss and optimizer")
    #chamfer_loss=ChamferLossSklearn()
    #chamfer_loss=ChamferLoss() # New: Added chamfer loss 
   # mean_squared_loss=torch.nn.MSELoss()
    #triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    # forward loop (training)
    losses=[]
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epochs):
            print(f'i am in the epochs')
            loss_total=0
            for index,data in enumerate(loader):
                print(f'index is {index}')
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
                ######### compute chamfer distance between source and target ########
                source_orig=torch.reshape(source,(source.shape[2],source.shape[0],source.shape[1]))
                target_orig=torch.reshape(target,(target.shape[2],target.shape[0],target.shape[1]))
                source_orig=source_orig.to(device)
                target_orig=target_orig.to(device)
                print(f'shape of source before feeding to chamfer function is {source_orig.shape}')
                print(f'shape of target before feeding to chamfer function is {target_orig.shape}')
                loss_chamfer_orig=chamfer_distance_sklearn(source_orig,target_orig)
                loss_chamfer_orig=loss_chamfer_orig.to(device)
                ###################################################################
                ########## forward pass of the model ##############################
                print("Shapes of source and target just before feeding into model")
                print(f'shape of source point cloud is {source.shape}')
                print(f'shape of target point cloud is {target.shape}')
                src_out_net,tgt_out_net = model(source,target)
                src_out_net=torch.squeeze(src_out_net)
                tgt_out_net=torch.squeeze(tgt_out_net)
                path_to_overlapping_indices=r"/scratch/fsa4859/OverlapPredator/overlap_pairs_modified_new/" +f"pair {index}" + ".npy"
                overlapping_indices=np.load(path_to_overlapping_indices,allow_pickle=True)
                overlapping_indices.astype(dtype=int,copy=False)
                overlapping_indices_torch=torch.from_numpy(overlapping_indices)
                overlapping_indices_torch=overlapping_indices_torch.to(device,dtype=torch.int64)
                # initialize ground truth for both source and target
                gt_src=torch.zeros(source.shape[0],requires_grad=True).to(device)
                gt_tgt=torch.zeros(target.shape[0],requires_grad=True).to(device)
                # select the indices of the inliers from overlap torch and set the value to 1 in the target
                gt_src[overlapping_indices_torch[:,0]]=1
                gt_tgt[overlapping_indices_torch[:,1]]=1
                ones_src=torch.nonzero(gt_src)
                ones_tgt=torch.nonzero(gt_tgt)
                print(f'number of inliers in grouund truth source is {ones_src.shape}')
                print(f'number of inliers in grouund truth target is {ones_tgt.shape}')
                # compute the loss
                #loss_src=BCE_Loss(src_out_net,gt_src)
                print(f'ground truth source is {gt_src}')
                print(f'result from model is {src_out_net}')
                print(f'shape of model output is {src_out_net.shape}')
                print(f'shape of source ground truth is {gt_src.shape}')
                print(f'shape of target ground truth is {gt_tgt.shape}')
                ########### compute the BCE for the source and target seperatlely ##########
                loss_src=BCE_Loss(src_out_net,gt_src)
                print(f'loss for src masking is {loss_src}')
                loss_tgt=BCE_Loss(tgt_out_net,gt_tgt)
                print(f'loss for target masking is {loss_tgt}')
                #############TODO: create versions of source and target for chamfer loss##########
                #source_chamfer=torch.reshape(source,(source.shape[2],source.shape[0],source.shape[1]))
                #source_chamfer=source_chamfer.to(device)
                #target_chamfer=torch.reshape(target,(target.shape[2],target.shape[0],target.shape[1]))
                #target_chamfer=target_chamfer.to(device)
                ##########TODO: select only the indices based on model predictions #########
                ######## find maximum and minimum of source out net and target out net ########
                max_source=torch.max(src_out_net)
                src_cond=src_out_net>=0.95*max_source # might need to change this value.
                #indices_src_out=src_cond.nonzero() # need to change this to make it work.
                indices_src_out=torch.where(src_out_net>=0.5,1,0)
                print(f'test tensor is {indices_src_out} and has shape of {indices_src_out.shape}')
                #indices_src_out=src_out_net[src_out_net>=0.5]
                indices_src_out=indices_src_out.nonzero()
                indices_src_out=torch.squeeze(indices_src_out)
                print(f'indices source out is {indices_src_out} and its dtype is {indices_src_out.dtype}')
                print(f'shape of inliers indices from source output is {indices_src_out.shape}')
                source_chamfer_final=source_orig[:,indices_src_out,:]
                ######### TODO: Repeat the same process for the target ########
                max_target=torch.max(tgt_out_net)
                #indices_tgt_out=torch.where(tgt_out_net>=0.5,tgt_out_net)  # new 
                tgt_cond=tgt_out_net>=0.95*max_target # might need to change this value. 
                indices_tgt_out=torch.where(tgt_out_net>=0.5,1,0)
                indices_tgt_out=indices_tgt_out.nonzero()
                #indices_tgt_out=tgt_out_net[tgt_out_net>=0.5]
                indices_tgt_out=torch.squeeze(indices_tgt_out)
                print(f'shape of inliers indices from source output is {indices_tgt_out.shape}')
                target_chamfer_final=target_orig[:,indices_tgt_out,:]
                print(f'shape of source model before feeding to chamfer function is {source_chamfer_final.shape}')
                print(f'shape of target model before feeding to chamfer function is {target_chamfer_final.shape}')
                loss_chamfer_predictions=chamfer_distance_sklearn(source_chamfer_final,target_chamfer_final)
                loss_chamfer_predictions=loss_chamfer_predictions.to(device)
                print("comparing chamfer distances between source and target")
                print(f'chamfer loss value between source and target model output is {loss_chamfer_predictions}')
                print(f'chamfer distance between transformed source and tgt is {loss_chamfer_orig}')
                print(f'shape of source is {source.shape}')
                print(f'shape of target is {target.shape}')
                #loss_chamfer=chamfer_distance_sklearn(source,target)
                #print(f'chamfer loss is {loss_chamfer} and type is {type(loss_chamfer)}')
                #print(f'checking if loss_chamfer is on cuda {loss_chamfer.is_cuda}')
                #loss_chamfer=loss_chamfer.to(device)
                if loss_chamfer_predictions<loss_chamfer_orig:
                    print("predictions consistency is better than original")
                    loss_chamfer_final=torch.tensor([0.0])
                elif loss_chamfer_predictions==loss_chamfer_orig:
                    print("two chamfer losses are equal")
                    loss_chamfer_final=loss_chamfer_orig
                else:
                    print("consistency of predicitons is less than original")
                    loss_chamfer_final=loss_chamfer_predictions-loss_chamfer_orig
                #loss_chamfer_final=min(loss_chamfer_predictions,loss_chamfer_orig)
                loss_chamfer_final=loss_chamfer_final.to(device)
                print(f'loss chamfer final is {loss_chamfer_final}')
                loss_src_tgt=loss_src+loss_tgt+loss_chamfer_final # New: Added the chamfer loss
                #### TODO: Add the chamfer loss function from sklearn #########
                # optimizer
                optimizer.zero_grad()
                loss_src_tgt.backward()
                optimizer.step()
                loss_total=loss_total+loss_src_tgt
            losses.append(loss_total/200)
            print(f"epoch {e}: loss:{loss_total/200}")
        
    torch.save(model,'/scratch/fsa4859/OverlapPredator/inlier_classification_model_abs/model.pth')

    
    


    





    
     




