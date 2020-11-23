# Cyclic Learning Rate

To determine the ideal max and min learning rates for CLR, an exponential learning rate test is performed. The learning is gradually increased and the loss is recorded. Then using the steps as highlighted in the thesis the ideal rates for the system can be determined.

## AbDataset.py

Defines the class to allow pytorch to obtain the images as required during training and testing. This is done as loading all the images would not be possible due to the sheer size

## Wavefront_CNN.py

This defines the network using pytorch. The code is self explanatory and further details can be obtained from the pytorch documentation.

## How to Use

To run, execute SensorModel_LR.py, adjusting all the paths as neccessary in AbDataset.py. Note, the results are recorded using tensorboard so ensure that is installed on the system. 