# Automated Capture

This code captures raw image at set intervals. The interval can be easily changed by changing the condition of the while loop

```python
while time() - start <= 4 * i:
```
In this example the interval is set to 4 seconds. Note the interval must be larger than the time taken to capture an image.

The code saves the raw data stream which can then be proccessed by the code presented in the Raw_Dataset folder.