import picamera
import picamera.array
import numpy as np
import io
from time import sleep, time


stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.framerate = 25
    camera.color_effects = (128, 128)
    camera.start_preview(resolution=(410, 313),
                         fullscreen=False, window=(20, 20, 820, 616))
    sleep(2)
    input('Good Luck')
    start = time()
    for i in range(1000):
        while time() - start <= 4 * i:
            pass
        print(time()-start)
        camera.capture(stream, 'jpeg', bayer=True)

        with open('PATH/Image{0}.data'.format(i), 'wb') as f:
            f.write(stream.getvalue())
        stream.seek(0)
        stream.truncate()
