import numpy as np
import torch

source=np.random.rand(1000,3)
target=np.random.rand(2000,3)
source_trans=torch.from_numpy(source)
target_trans=torch.from_numpy(target)
l2_dist=torch.cdist(source_trans,target_trans)
print(l2_dist) # shape is (1000,2000) each row represents the distance between one point in the source to all points in target
threshold = 15 * torch.min(l2_dist)
overlap_indexes = torch.nonzero(l2_dist < threshold) # this will generate two dimensional tensor.
print(overlap_indexes) # shape of the overlapping indices is (number_indices,2)
gt = torch.zeros(source_trans.shape[0], 2) # shape is (1000,2)
gt[overlap_indexes[:, 0], 0] = 1 # getting all source indices and setting them to 1
gt[overlap_indexes[:, 0], 1] = 1 # getting all source indices and setting them to 1
print(gt)