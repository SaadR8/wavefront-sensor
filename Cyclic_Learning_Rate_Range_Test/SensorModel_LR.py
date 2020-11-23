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
from LearningRate import cyclical_lr
import math

writer = SummaryWriter()

transform = T.Compose(
    [T.Grayscale(num_output_channels=1),
     T.CenterCrop([1024, 1024]),
     T.ToTensor(),
     T.Normalize([0.1599], [0.1404])])

params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 0,
          'pin_memory': True}


"""
test = Image.open("Image0.tif")
test = TF.affine(test, -6, (-50, 0), 1, 0)

test = transform(test)
test.show()
"""


dataset = AbDataset(transform)

train_set, test_set = random_split(dataset, [800, 200])

train_generator = DataLoader(train_set, **params)

test_generator = DataLoader(test_set, **params)

net = Net(1, 1024, 1024).cuda()

#writer.add_graph(net, torch.rand(1, 1, 1024, 1024).cuda())

epochs = 4


start_lr = 1e-7
end_lr = 0.05

criterion = nn.MSELoss()
optimiser = optim.SGD(net.parameters(), start_lr)


def lr_lambda(x): return math.exp(x * math.log(end_lr / start_lr) /
                                  (epochs * len(train_set)/2))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

lr_find_loss = []
lr_find_lr = []

iteration = 0

smoothing = 0.05

for epoch in range(epochs):
    #running_loss = 0.0
    for i, data in enumerate(train_generator):

        pattern, zernike = data

        optimiser.zero_grad()

        outputs = net(pattern.cuda().float())
        loss = criterion(outputs, zernike.cuda().float())
        loss.backward()
        optimiser.step()

        scheduler.step()
        lr_step = optimiser.state_dict()["param_groups"][0]["lr"]
        lr_find_lr.append(lr_step)
        writer.add_scalar('step', lr_step, iteration)
        if iteration == 0:
            lr_find_loss.append(loss)
            writer.add_scalar('loss', loss.item(), iteration)
        else:
            loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
            lr_find_loss.append(loss)
            writer.add_scalar('loss', loss.item(), iteration)

        iteration += 1
        print(iteration)
        """
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))

            #writer.add_scalar('training loss', running_loss /2000, epoch * len(train_generator)+i)
            running_loss = 0.0
        """


writer.close()

#PATH = '/data/engs-imaging-disorder/kebl5493/Sensor_NetLR.pth'
#torch.save(net.state_dict(), PATH)


"""
correct = [0.0, 0.0, 0.0, 0.0, 0.0]
total = [0.0, 0.0, 0.0, 0.0, 0.0]
runLoss = [0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in test_generator:
        pattern, zernike = data
        zernike = zernike.tolist()
        output = net(pattern.cuda().float())
        output = output.cpu().tolist()

        for i in range(2):
            for j in range(5):
                if abs(zernike[i][j] - output[i][j]) < 0.04:
                    correct[j] += 1
                runLoss[j] += abs(zernike[i][j] - output[i][j])
            total[zernike[i].index(max(zernike[i]))] += 1

for k in range(5):
    writer.add_scalar('percentage correct', (correct[k]/sum(total)) * 100, k+1)
    writer.add_scalar('average loss', runLoss[k]/sum(total), k+1)


writer.close()

for l in range(5):
    print((correct[l]/sum(total))*100, runLoss[l] /
          sum(total), total[l], sep="---")

"""
