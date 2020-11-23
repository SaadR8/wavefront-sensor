import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from AbDataset import AbDataset
from Wavefront_CNN import Net
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF
import time

writer = SummaryWriter()

transform = T.Compose(
    [T.Grayscale(num_output_channels=1),
     T.CenterCrop([1024, 1024]),
     T.ToTensor(),
     T.Normalize([0.1599], [0.1404])])

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0,
          'pin_memory': True}


dataset = AbDataset(transform)

data_generator = DataLoader(dataset, **params)


net = Net(1, 1024, 1024).cuda()

epochs = 10

criterion = nn.MSELoss()
optimiser = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(epochs):
    running_loss = 0.0
    t0 = time.time()
    for i, data in enumerate(data_generator):

        pattern, zernike = data

        optimiser.zero_grad()

        outputs = net(pattern.cuda().float())
        loss = criterion(outputs, zernike.cuda().float())
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))

            writer.add_scalar('training loss', running_loss /
                              2000, epoch * len(data_generator)+i)
            running_loss = 0.0

    print(time.time() - t0)

print("ADAM Complete")

writer.close()


#PATH = '/home/kebl5493/Sensor_Net2.pth'
#torch.save(net.state_dict(), PATH)
