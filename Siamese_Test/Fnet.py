import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FNet(nn.Module):
    def __init__(self, dimx, dimy):
        super(FNet, self).__init__()

        self.fc1 = nn.Linear(1*dimx*dimy, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 5)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
