import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
from demons_registration import multiscale_demons
from PIL import Image

import matlab.engine
mat = matlab.engine.start_matlab()


def command_iteration(filter):
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                     filter.GetMetric()))


fixed = Image.open('Path to zero aberration image')
moving = Image.open('Path to aberrated image')

fixed = TF.affine(fixed, -6, (-50, 0), 1, 0)
moving = TF.affine(moving, -6, (-50, 0), 1, 0)

transform = T.Compose(
    [T.Grayscale(num_output_channels=1),
     T.CenterCrop([720, 720])])

fixed = transform(fixed)
moving = transform(moving)


fixed = sitk.GetImageFromArray(fixed)
moving = sitk.GetImageFromArray(moving)


matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(256)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving = matcher.Execute(moving, fixed)


demons = sitk.DemonsRegistrationFilter()
demons.SetNumberOfIterations(100)
demons.SetSmoothDisplacementField(True)
demons.SetStandardDeviations(1.0)

demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))


displacementField = multiscale_demons(registration_algorithm=demons, fixed_image=fixed,
                                      moving_image=moving, shrink_factors=[16, 8, 4, 2], smoothing_sigmas=[1, 1, 1, 1])


df = sitk.GetArrayFromImage(displacementField.GetDisplacementField())

dfx, dfy = np.dsplit(df, 2)
dfx = np.squeeze(dfx)
dfy = np.squeeze(dfy)

#sci.savemat("grad.mat", {'dfx': dfx, 'dfy': dfy})

dfx = (dfx * -(1.12/2200))
dfy = (dfy * -(1.12/2200))


dfx = matlab.double(dfx.tolist())
dfy = matlab.double(dfy.tolist())

Phi = mat.intgrad2(dfx, dfy, 1.12, 1.12)

Phi = np.array(Phi)

#plt.imshow(Phi, interpolation='nearest')
# plt.show()

np.save("Phi", Phi)


"""
Code to obatain the transformed image

outTx = sitk.DisplacementFieldTransform(displacementField)
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(outTx)

out = resampler.Execute(moving)
t = sitk.GetArrayFromImage(out)
t = Image.fromarray(t)
t.save("transformed.tif")

f = sitk.GetArrayFromImage(fixed)
f = Image.fromarray(f)
f.save("fixed.tif")

m = sitk.GetArrayFromImage(moving)
m = Image.fromarray(m)
m.save("moving.tif")
"""
