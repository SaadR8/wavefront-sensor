# Siamese Network

Here the code uses the relative measurement, this requires the feature extraction network (U-Net) and the classifying network to be separated.

## Unet2.py

This contains the U-Net model with swish activation function in pytorch.

## Fnet.py

This contains the classifying network which are 3 fully connnected layers.

## AbDataset.py

Defines the class to allow pytorch to obtain the images as required during training and testing. This is done as loading all the images would not be possible due to the sheer size

## Wavefront_CNN.py

This defines the network using pytorch. The code is self explanatory and further details can be obtained from the pytorch documentation.

## How to Use

To run, execute SensorModel_Siamese.py, adjusting all the paths as neccessary in AbDataset.py. Note, the results are recorded using tensorboard so ensure that is installed on the system. 