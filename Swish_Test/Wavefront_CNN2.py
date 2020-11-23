import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def swish(x):
    return x * torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self, ch, dimx, dimy):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=64,
                               kernel_size=3, stride=1, padding=1,)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1,)

        self.norm1 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1,)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1,)

        self.norm2 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(2, 2, 0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1,)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1, padding=1,)

        self.norm3 = nn.BatchNorm2d(256)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=1, padding=1,)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1,)

        self.norm4 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=1, padding=1,)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1,)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=1,
                                kernel_size=3, stride=1, padding=1,)

        self.norm5 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(ch*dimx*dimy, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 5)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = self.norm1(x)

        res1 = x

        x = self.maxpool1(x)
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.norm2(x)

        res2 = x

        x = self.maxpool2(x)
        x = swish(self.conv5(x))
        x = swish(self.conv6(x))
        x = self.norm3(x)

        x = self.upconv1(x)
        x = torch.cat((res2, x), 1)
        x = swish(self.conv7(x))
        x = swish(self.conv8(x))
        x = self.norm4(x)

        x = self.upconv2(x)
        x = torch.cat((res1, x), 1)
        x = swish(self.conv9(x))
        x = swish(self.conv10(x))
        x = swish(self.conv11(x))
        x = self.norm5(x)

        x = x.view(x.size(0), -1)
        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        x = self.fc3(x)

        return x
