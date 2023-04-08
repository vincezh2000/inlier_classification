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
        # get resnet model
        #self.resnet = torchvision.models.resnet18(weights=None)
        self.feature_extractor=PointNet()

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        #self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        #self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        #self.fc = nn.Sequential(
            #nn.Linear(self.fc_in_features * 2, 256),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, 1),
        #)

        #self.fc = nn.Sequential(
            #nn.Linear(100, 256),
            #nn.ReLU(inplace=True),
            #nn.Linear(256, 1))
        
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
        output1 = self.forward_once(input1)
        print(f'shape of output 1 after DCP  is {output1.shape}') # output is of shape (pts,64,1)
        output2 = self.forward_once(input2)
        print(f'shape of output 2 after DCP is {output2.shape}') # output is of shape (pts,64,1)
        # TODO: fix the concatenation: April 8,2023
        ############ concatenate target max pooled features to source to create source mask ################
        output_2_max=torch.max(output2,dim=0)
        output_2_max_new=output_2_max[0]
        target_maxPooled_duplicated=output_2_max_new.repeat(input1.shape[0],1,1) # shape (pts,64,1)
        print(f'shape of target max pooled after duplicating is {target_maxPooled_duplicated.shape}')
        ############ concatenate source and target in the feature dimension (dim=1) ##################
        global_source=torch.cat((output1,target_maxPooled_duplicated),dim=1)  # shape (pts,128,1)
        print(f'shape of global feature vector  after concatenating is {global_source.shape}')
        global_source=global_source.reshape((global_source.shape[0],global_source.shape[2],global_source.shape[1]))
        print(f'shape of global feature vector before fully connected layer {global_source.shape}')
        # TODO: fix the concatenation to concatenate source and t
        #output2_max=torch.max(output2,dim=0)
        #output2_max_tensor=output2_max[0]
        #output2_max_new=torch.reshape(output2_max_tensor,(1,output2_max_tensor.shape[0],1))
        # TODO: now concatentate the source and target
        #output_src=torch.cat((output1,output2_max_new),dim=0)
        #output = torch.cat((output1, output2), 1)
        #output_src=output_src.reshape((output_src.shape[0],output_src.shape[2],output_src.shape[1]))
        #print(f'shape of output after concatenating is {output_src.shape}')

        # TODO: add the target mask
        #output1_max=torch.max(output1,dim=0)
        #output1_max_tensor=output1_max[0]
        #output1_max_new=torch.reshape(output1_max_tensor,(1,output1_max_tensor.shape[0],1))
        #output_tgt=torch.cat((output2,output1_max_new),dim=0)
        #output_tgt=output_tgt.reshape((output_tgt.shape[0],output_tgt.shape[2],output_tgt.shape[1]))

        # TODO: Modify the shape again so that ground truth works
        #print(f'shape of output modified just before reshaping is {output_src.shape}')
        #output_modified=torch.reshape(output,(output.shape[2],output.shape[1],output.shape[0]))
        #print(f'shape of output modified just before linear layer is {output_src.shape}')
        #print(f'printing shape of input point cloud {input1.shape}')
        #linear_layer=nn.Linear(output_modified.shape[2],input1.shape[0]).to(device)
        #output_modified_final=linear_layer(output_modified)
        #print(f'shape of output before fully connected layer is {output_modified.shape}')
        #output_modified_final=torch.reshape(output_modified_final,(output_modified_final.shape[2],output_modified_final.shape[1],output_modified_final.shape[0]))
        # pass the concatenation to the linear layers
        mask_src = self.fc(global_source)
        #mask_tgt=self.fc(output_tgt)
        print(f'shape of output after fully connected is {mask_src.shape}')
        # pass the out of the linear layers to sigmoid layer
        mask_src = self.sigmoid(mask_src)
        #mask_tgt=self.sigmoid(mask_tgt)
        print(f'shape of output after sigmoid activation is {mask_src.shape}')

        #print(f'shape of final output is {output.shape}')

        #print(f'output after sigmoid is {output}')
        return mask_src


