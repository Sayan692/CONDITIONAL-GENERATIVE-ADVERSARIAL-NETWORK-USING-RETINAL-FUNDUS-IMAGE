import cv2
import os
import numpy as np

for (root,dirs,files) in os.walk("/home/rayuga/Documents/DataSet/GAN/256x256_test"):
    for f in files:
        mask=np.zeros(shape=[256,256],dtype=np.uint8)
        img=cv2.imread(root+'/'+f)
        mask[img[:,:,2]>50]=255
        #img=cv2.resize(img,(256,256))
        cv2.imwrite(os.path.join("/home/rayuga/Documents/DataSet/GAN/256x256_mask",f),mask)
        #print("saving")
