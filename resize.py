import cv2
import os

for (root,dirs,files) in os.walk("/home/rayuga/Documents/DataSet/GAN/original_data/PNG"):
    for f in files:
        img=cv2.imread(root+'/'+f)
        img=cv2.resize(img,(256,256))
        cv2.imwrite(os.path.join("/home/rayuga/Documents/DataSet/GAN/minor_data/256x256_original",f),img)
        