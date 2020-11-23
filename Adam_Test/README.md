# ADAM Test

This code was used to test the effectiveness of the ADAM optimiser for training the network. The learning rate can be changed in the following line in SensorModel_Adam.py

```python
optimiser = optim.Adam(net.parameters(), lr=0.001)
```

## AbDataset.py

Defines the class to allow pytorch to obtain the images as required during training and testing. This is done as loading all the images would not be possible due to the sheer size

## Wavefront_CNN.py

This defines the network using pytorch. The code is self explanatory and further details can be obtained from the pytorch documentation.

## How to Use

To run, execute SensorModel_Adam.py, adjusting all the paths as neccessary in AbDataset.py. Note, the results are recorded using tensorboard so ensure that is installed on the system. 