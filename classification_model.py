import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_shape,1000)
        self.fc2 = nn.Linear(1000,800)
        self.fc3 = nn.Linear(800,500)
        self.fc4=nn.Linear(500,1000)
        self.fc5=nn.Linear(1000,500)
        self.fc6=nn.Linear(500,2)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.softmax(self.fc6(x),dim=1)
        return x

