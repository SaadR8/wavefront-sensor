import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from AbDataset import AbDataset
from Wavefront_CNN2 import Net
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF
from LearningRate2 import cyclical_lr
import time

writer = SummaryWriter()

transform = T.Compose(
    [T.Grayscale(num_output_channels=1),
     T.CenterCrop([1024, 1024]),
     T.ToTensor(),
     T.Normalize([0.1599], [0.1404])])

params_train = {'batch_size': 2,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': True}

params_test = {'batch_size': 2,
               'shuffle': True,
               'num_workers': 0,
               'pin_memory': True}

dataset = AbDataset(transform)

train_set, test_set = random_split(dataset, [800, 200])

train_generator = DataLoader(train_set, **params_train)

test_generator = DataLoader(test_set, **params_test)

net = Net(1, 1024, 1024).cuda()

epochs = 10

criterion = nn.MSELoss()
optimiser = optim.SGD(net.parameters(), lr=1.)
step_size = 4 * len(train_generator)
clr = cyclical_lr(step_size)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, [clr])


for epoch in range(epochs):
    running_loss = 0.0
    t0 = time.time()
    for i, data in enumerate(train_generator):

        pattern, zernike = data

        optimiser.zero_grad()

        outputs = net(pattern.cuda().float())
        loss = criterion(outputs, zernike.cuda().float())
        loss.backward()
        optimiser.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))

            writer.add_scalar('training loss', running_loss /
                              10, epoch * len(train_generator)+i)
            running_loss = 0.0

    print(time.time() - t0)

print("Swish Complete")

writer.close()

Tpos = [0.0, 0.0, 0.0, 0.0, 0.0]
Tfal = [0.0, 0.0, 0.0, 0.0, 0.0]
total = [0.0, 0.0, 0.0, 0.0, 0.0]
runLossPos = [0.0, 0.0, 0.0, 0.0, 0.0]
runLossFal = [0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in test_generator:
        pattern, zernike = data
        zernike = zernike.tolist()
        output = net(pattern.cuda().float())
        output = output.cpu().tolist()

        for i in range(2):
            posIndex = zernike[i].index(max(zernike[i]))
            for j in range(5):
                if abs(zernike[i][j] - output[i][j]) < 0.05 and posIndex == j:
                    Tpos[j] += 1
                    runLossPos[j] += abs(zernike[i][j] - output[i][j])
                elif abs(zernike[i][j] - output[i][j]) < 0.05:
                    Tfal[j] += 1
                    runLossFal[j] += abs(zernike[i][j] - output[i][j])
            total[zernike[i].index(max(zernike[i]))] += 1

print(*Tpos, sep="---")
print(*Tfal, sep="---")
print(*total, sep="---")
print(*runLossPos, sep="---")
print(*runLossFal, sep="---")
