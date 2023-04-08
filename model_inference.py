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
import open3d as o3d

if __name__=='__main__':
    if torch.cuda.is_available():
        dev="cuda:0"
    else:
        dev="cpu"
    device=torch.device(dev)
    learning_rate = 0.001 # try the new learning rate + double check the softmax results + input visualization Aditya + check dcp code (ex: batch norm layer) 
    epochs = 20
    CFG_DIR = r"/scratch/fsa4859/OverlapPredator/configs/train/kitti.yaml"
    config = edict(load_config(CFG_DIR))
    logging.warning("Starting the data loading process")
    dataset = KITTIDataset(config=config,split="train", data_augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'the length of the data loader is {len(loader)}')
    inference_dir=os.makedirs("model_inference",exist_ok=True)
    model=torch.load('/scratch/fsa4859/OverlapPredator/inlier_classification_model/model.pth')
    model.to(torch.device("cuda:0"))
    model.eval()
    with torch.autograd.set_detect_anomaly(True):
        for index,data in enumerate(loader):
                if index>50:
                    break
                source=data[0].float().squeeze(0)
                target=data[1].float().squeeze(0)
                rot=data[4].float().squeeze(0)
                trans=data[5].float().squeeze(0)
                target=target.reshape((target.shape[0],target.shape[1],1))
                source=source.reshape((source.shape[0],source.shape[1],1))
                if source.shape[0]!=target.shape[0]:
                    if source.shape[0]>target.shape[0]:
                        source=source[0:target.shape[0],:,:]
                    else:
                        target=target[0:source.shape[0],:,:]
                print(f'shape of source point cloud is {source.shape}')
                print(f'shape of target point cloud is {target.shape}')
                source_transformed=((rot@source.T)+trans).T
                ######### Running ICP ############
                trans_init=np.eye(4)
                source_transformed_np=source_transformed.numpy()
                tgt_np=target.numpy()
                source_transformed_pcd = o3d.geometry.PointCloud()
                source_transformed_pcd.points=o3d.utility.Vector3dVector(source_transformed_np)
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points=o3d.utility.Vector3dVector(tgt_np)
                source_transformed_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05*2, max_nn=200))
                tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05*2, max_nn=200))
                reg_p2p = o3d.pipelines.registration.registration_icp(source_transformed_pcd, tgt_pcd, 0.02, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                print(reg_p2p)
                print("Transformation is:")
                print(reg_p2p.transformation)
                print("Correspondence set:")
                print(reg_p2p.correspondence_set)
                print("indices of correspondences:")
                inlier_indices=np.asarray(reg_p2p.correspondence_set)
                print(inlier_indices)
                print(f'shape of inliers indices: {inlier_indices.shape}')
                source=source.to(device)
                target=target.to(device)
                #logging.warning("Loading data into model")
                src_out_net = model(source,target)
                src_out_net=torch.squeeze(src_out_net)
                src_out_numpy=src_out_net.detach().cpu().numpy()
                print(f'printing output from model: {src_out_net}')
                print(f'printing shape of model output {src_out_net.shape}')
                max_value,max_indices=torch.max(src_out_net,dim=0)
                min_value,min_indices=torch.min(src_out_net,dim=0)
                int_min=int(min_value)
                print(f'type of source out net is {type(src_out_net)}')
                condition_tensor=src_out_net>0.9
                int_tensor=condition_tensor.to(dtype=torch.int64)
                print(f'int tensor is {int_tensor}')
                #num_inliers=(int_tensor == 1).sum(dim=0)
                num_inliers=((int_tensor == 1).nonzero(as_tuple=True)[0])
                print(f'number of inliers is {num_inliers}')
                print(f'shape of inliers is {num_inliers.shape}')
                #indices_max=torch.where(src_out_net>int_min,src_out_net)
                print(f'printing max element in model outout {max_value}')
                print(f'printing min element in model output {min_value}')
                print(f'printing indices of max value {max_indices}')
                print(f'printing indices of min in model output {min_indices}')
                inference_dir=os.path.join("model_inference",f"pair {index}.npy")
                np.save(inference_dir,src_out_numpy)




      