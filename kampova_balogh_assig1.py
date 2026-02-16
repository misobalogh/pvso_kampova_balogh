from ximea import xiapi
import cv2
import os
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial numBber)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()

img_dir = "imgs"
os.makedirs(img_dir, exist_ok=True)


width = 240
height = 240
num_channels = 4
mosaic = np.zeros((2*width, 2*height, num_channels))

taken_pics = 0
while cv2.waitKey() != ord('q') and taken_pics < 4:
    
    k = cv2.waitKey()
    
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(width,height))

    if k == 32:
        taken_pics += 1
        cv2.imwrite(f"{img_dir}/image_{taken_pics}.jpeg", image)


    num_channels = image.shape[2]

    mosaic[taken_pics*width:width+taken_pics*width, 
           taken_pics*height:height+taken_pics*height,] += image

    cv2.imshow("test", mosaic)
    
       

# for i in range(4):
#     #get data and pass them from camera to img
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     cv2.imshow("test", image)
#     cv2.waitKey()
#     #get raw data from camera
#     #for Python2.x function returns string
#     #for Python3.x function returns bytes
#     data_raw = img.get_image_data_raw()

#     #transform data to list
#     data = list(data_raw)

#     #print image data and metadata
#     print('Image number: ' + str(i))
#     print('Image width (pixels):  ' + str(img.width))
#     print('Image height (pixels): ' + str(img.height))
#     print('First 10 pixels: ' + str(data[:10]))
#     print('\n')

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')