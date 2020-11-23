import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from AbDataset import AbDataset
from Unet2 import UNet
from Fnet import FNet
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF
from LearningRate2 import cyclical_lr

writer = SummaryWriter()

transform = T.Compose(
    [T.Grayscale(num_output_channels=1),
     T.CenterCrop([1024, 1024]),
     T.ToTensor(),
     T.Normalize([0.1599], [0.1404])])

params_train = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': True}

params_test = {'batch_size': 1,
               'shuffle': True,
               'num_workers': 0,
               'pin_memory': True}


ref = Image.open("/data/engs-imaging-disorder/kebl5493/dataset2/Image0.tif")
#ref = Image.open("Image0.tif")
ref = TF.affine(ref, -6, (-50, 0), 1, 0)
ref = transform(ref).unsqueeze(0).cuda().float()
#ref = torch.cat((ref, ref), 0).cuda().float()


dataset = AbDataset(transform)

train_set, test_set = random_split(dataset, [800, 200])

train_generator = DataLoader(train_set, **params_train)

test_generator = DataLoader(test_set, **params_test)


unet = UNet(1, 1024, 1024).cuda()
fnet = FNet(1024, 1024).cuda()

# writer.add_graph(net, torch.rand(1, 1, 1024, 1024).cuda())

epochs = 10

params = [x for x in unet.parameters()] + [x for x in fnet.parameters()]

criterion = nn.MSELoss()
optimiser = optim.SGD(params=params, lr=1.)
step_size = 4 * len(train_generator)
clr = cyclical_lr(step_size)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, [clr])

# w.add_graph(net, train_set)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_generator):

        unet.train()
        fnet.train()

        pattern, zernike = data

        unet.zero_grad()
        fnet.zero_grad()

        output1 = unet(ref)
        output2 = unet(pattern.cuda().float())

        #output = fnet(torch.cat((output1, output2), 1))
        output = fnet((output1 - output2))

        loss = criterion(output, zernike.cuda().float())
        loss.backward()
        optimiser.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))

            writer.add_scalar('training loss', running_loss /
                              2000, epoch * len(train_generator)+i)
            running_loss = 0.0

writer.close()

print("Siamese Complete")
#PATH1 = '/home/kebl5493/Unet.pth'
#PATH2 = '/home/kebl5493/Fnet.pht'
#torch.save(unet.state_dict(), PATH1)
#torch.save(fnet.state_dict(), PATH2)


Tpos = [0.0, 0.0, 0.0, 0.0, 0.0]
Tfal = [0.0, 0.0, 0.0, 0.0, 0.0]
total = [0.0, 0.0, 0.0, 0.0, 0.0]
runLossPos = [0.0, 0.0, 0.0, 0.0, 0.0]
runLossFal = [0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in test_generator:
        pattern, zernike = data
        zernike = zernike.cpu().tolist()

        output1 = unet(ref)
        output2 = unet(pattern.cuda().float())

        output = fnet((output1 - output2)).cpu().tolist()

        for i in range(1):
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
