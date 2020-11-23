# Swish Activation Function

Here the ReLU activation function is replaced with the swish function which is now stored in Wavefront_CNN2.py

## LearningRate.py

This tells pytorch how to alter the learning rate at each step. The max and min learning rate can be easily configured by altering the default values in the function definition.

```python
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
```

## AbDataset.py

Defines the class to allow pytorch to obtain the images as required during training and testing. This is done as loading all the images would not be possible due to the sheer size

## Wavefront_CNN2.py

This defines the network using pytorch. The code is self explanatory and further details can be obtained from the pytorch documentation.

## How to Use

To run, execute SensorModel_Adam.py, adjusting all the paths as neccessary in AbDataset.py. Note, the results are recorded using tensorboard so ensure that is installed on the system. 