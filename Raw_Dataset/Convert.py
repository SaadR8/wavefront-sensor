import io
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import picamraw


# Extract the raw Bayer data from the end of the stream, check the
# header and strip if off before converting the data into a numpy array

for i in range(1000):

    stream = open('E:\dataset\Image{0}.data'.format(i), "rb")
    stream = io.BytesIO(stream.read())
    ver = 2

    offset = {
        1: 6404096,
        2: 10270208,
    }[ver]

    # Extract the raw Bayer data from the end of the stream, check the
    # header and strip if off before converting the data into a numpy array

    offset = {
        1: 6404096,
        2: 10270208,
    }[ver]
    data = stream.getvalue()[-offset:]
    assert data[:4] == b'BRCM'
    data = data[32768:]
    data = np.fromstring(data, dtype=np.uint8)

    # For the V1 module, the data consists of 1952 rows of 3264 bytes of data.
    # The last 8 rows of data are unused (they only exist because the maximum
    # resolution of 1944 rows is rounded up to the nearest 16).
    #
    # For the V2 module, the data consists of 2480 rows of 4128 bytes of data.
    # There's actually 2464 rows of data, but the sensor's raw size is 2466
    # rows, rounded up to the nearest multiple of 16: 2480.
    #
    # Likewise, the last few bytes of each row are unused (why?). Here we
    # reshape the data and strip off the unused bytes.

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
    }[ver]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    # Horizontally, each row consists of 10-bit values. Every four bytes are
    # the high 8-bits of four values, and the 5th byte contains the packed low
    # 2-bits of the preceding four values. In other words, the bits of the
    # values A, B, C, D and arranged like so:
    #
    #  byte 1   byte 2   byte 3   byte 4   byte 5
    # AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD DDCCBBAA
    #
    # Here, we convert our data into a 16-bit array, shift all values left by
    # 2-bits and unpack the low-order bits from every 5th byte in each row,
    # then remove the columns containing the packed bits

    data = data.astype(np.uint16) << 2
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> (byte * 2)) & 0b11)
    data = np.delete(data, np.s_[4::5], 1)

    # Now to split the data up into its red, green, and blue components. The
    # Bayer pattern of the OV5647 sensor is BGGR. In other words the first
    # row contains alternating green/blue elements, the second row contains
    # alternating red/green elements, and so on as illustrated below:
    #
    # GBGBGBGBGBGBGB
    # RGRGRGRGRGRGRG
    # GBGBGBGBGBGBGB
    # RGRGRGRGRGRGRG
    #
    # Please note that if you use vflip or hflip to change the orientation
    # of the capture, you must flip the Bayer pattern accordingly

    ((ry, rx), (gy, gx), (Gy, Gx), (by, bx)) = ((1, 1), (0, 1), (1, 0), (0, 0))

    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[ry::2, rx::2, 0] = data[ry::2, rx::2]  # Red
    rgb[gy::2, gx::2, 1] = data[gy::2, gx::2]  # Green
    rgb[Gy::2, Gx::2, 1] = data[Gy::2, Gx::2]  # Green
    rgb[by::2, bx::2, 2] = data[by::2, bx::2]  # Blue """

    # At this point we now have the raw Bayer data with the correct values
    # and colors but the data still requires de-mosaicing and
    # post-processing. If you wish to do this yourself, end the script here!
    #
    # Below we present a fairly naive de-mosaic method that simply
    # calculates the weighted average of a pixel based on the pixels
    # surrounding it. The weighting is provided by a byte representation of
    # the Bayer filter which we construct first:

    #array_3d = np.load("test50.npy")

    array_3d = rgb

    bayer = np.zeros(array_3d.shape, dtype=np.uint8)

    bayer[ry::2, rx::2, 0] = 1  # Red
    bayer[gy::2, gx::2, 1] = 1  # Green
    bayer[Gy::2, Gx::2, 1] = 1  # Green
    bayer[by::2, bx::2, 2] = 1  # Blue
    # Allocate output array with same shape as data and set up some
    # constants to represent the weighted average window
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)
    # Pad out the data and the bayer pattern (np.pad is faster but
    # unavailable on the version of numpy shipped with Raspbian at the
    # time of writing)
    rgb = np.zeros((
        array_3d.shape[0] + borders[0],
        array_3d.shape[1] + borders[1],
        array_3d.shape[2]), dtype=array_3d.dtype)
    rgb[
        border[0]:rgb.shape[0] - border[0],
        border[1]:rgb.shape[1] - border[1],
        :] = array_3d
    bayer_pad = np.zeros((
        array_3d.shape[0] + borders[0],
        array_3d.shape[1] + borders[1],
        array_3d.shape[2]), dtype=bayer.dtype)
    bayer_pad[
        border[0]:bayer_pad.shape[0] - border[0],
        border[1]:bayer_pad.shape[1] - border[1],
        :] = bayer
    bayer = bayer_pad
    # For each plane in the RGB data, construct a view over the plane
    # of 3x3 matrices. Then do the same for the bayer array and use
    # Einstein summation to get the weighted average
    output = np.empty(array_3d.shape, dtype=array_3d.dtype)
    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]
        pview = as_strided(p, shape=(
            p.shape[0] - borders[0],
            p.shape[1] - borders[1]) + window, strides=p.strides * 2)
        bview = as_strided(b, shape=(
            b.shape[0] - borders[0],
            b.shape[1] - borders[1]) + window, strides=b.strides * 2)
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        output[..., plane] = psum // bsum

    # At this point output should contain a reasonably "normal" looking
    # image, although it still won't look as good as the camera's normal
    # output (as it lacks vignette compensation, AWB, etc).
    #
    # If you want to view this in most packages (like GIMP) you'll need to
    # convert it to 8-bit RGB data. The simplest way to do this is by
    # right-shifting everything by 2-bits (yes, this makes all that
    # unpacking work at the start rather redundant...)

    output = (output >> 2).astype(np.uint8)
    img = Image.fromarray(output)
    img.save('F:\data2\Image{0}.tif'.format(i))
