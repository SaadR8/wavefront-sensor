# Demons Algorithm

The demons algorithm allows the displacement field between the reference caustic and an aberrated caustic to be obtained. This can then be scaled and integrated to obtain the wavefront. Note the integration function is taken from MATLAB, therefore ensure that matlab engine for python is installed.

## demons_registration.py

This contains the code for the registration algortithm itself implemented using the SimpleITK library.

## Multi_Demons.py

This takes in the images and performs the registration using the functions in demons_registration.py. Extra code is also provided in the comments at the end to allow the transformed image to be obtained. The wavefront is stored in an numpy array and saved into a .npy file.

## ZExtraction.py

This loads the .npy file and extracts the zernike coefficents from the wavefront and plots the result in polar coordinates to more closely match the output of a Shack-Hartmann sensor.

## How to Use

Update the paths as required. Execute Multi_Demons then ZExtraction to obtain the zernike coefficents from an aberrated imagge.