import os
import numpy as np
import random
import math
import json
from functools import partial
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import copy
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

class PointNet(nn.Module):
    def __init__(self, emb_dims=64):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class mlp(nn.Module):
    '''
    Multi-layer perceptron class
    '''
    def __init__(self):
        super().__init__()
        #self.nn=nn.Flatten()
        self.layers=nn.Sequential(
           nn.Flatten(),
           nn.Linear(300,64),
           nn.ReLU(),
           nn.Linear(64,128),
           nn.ReLU(),
           nn.Linear(128,256),
           nn.ReLU(),
           nn.Linear(256,512),
           nn.ReLU(),
           nn.Linear(512,1024),
           nn.ReLU())
    def forward(self,x):
        return self.layers(x)

class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor=PointNet()
        
        self.fc=nn.Sequential(
            nn.Linear(128,1),
            nn.ReLU(inplace=True),
            nn.Linear(1,1)
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        #self.resnet.apply(self.init_weights)
        self.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        #output = self.resnet(x)
        output = self.feature_extractor(x)
        #output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two point cloud' features
        output1 = self.forward_once(input1) # output is of shape (pts,64,1)
        output2 = self.forward_once(input2) # output is of shape (pts,64,1)
        # TODO: fix the concatenation: April 8,2023
        ############ concatenate target max pooled features to source to create source mask ################
        output_2_max=torch.max(output2,dim=0)
        output_2_max_new=output_2_max[0]
        target_maxPooled_duplicated=output_2_max_new.repeat(input1.shape[0],1,1) # shape (pts,64,1)
        ############ concatenate source and target in the feature dimension (dim=1) ##################
        global_source=torch.cat((output1,target_maxPooled_duplicated),dim=1)  # shape (pts,128,1)
        global_source=global_source.reshape((global_source.shape[0],global_source.shape[2],global_source.shape[1]))
        ########### concatenate source max pooled features to targer to create target mask ###########
        output_1_max=torch.max(output1,dim=0)
        output_1_max_new=output_1_max[0]
        source_maxPooled_duplicated=output_1_max_new.repeat(input1.shape[0],1,1) # shape (pts,64,1)
        ############ concatenate source and target in the feature dimension (dim=1) ##################
        global_target=torch.cat((output2,source_maxPooled_duplicated),dim=1)  # shape (pts,128,1)
        global_target=global_target.reshape((global_target.shape[0],global_target.shape[2],global_target.shape[1]))
    
        ############ pass the feature vectors to fully connected layers ###########    
        mask_src = self.fc(global_source)
        mask_tgt=self.fc(global_target)
        # pass the out of the linear layers to sigmoid layer
        mask_src = self.sigmoid(mask_src)
        mask_tgt=self.sigmoid(mask_tgt)
        print(f'shape of output after sigmoid activation is {mask_src.shape}')
        return mask_src,mask_tgt


