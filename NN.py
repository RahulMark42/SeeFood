import os
import cv2
import pickle
import numpy as np
from HelperFunctions import forward_propagation_n

img = cv2.imread(file_path)

'''cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

def PreProcessImage(img):
    img_size = 150
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))
     
    img_rgb = np.array(img_rgb)
 
    img_rgb = img_rgb/255.0
       
    img_reshape = img_rgb.reshape(img_rgb.shape[0] * img_rgb.shape[1] * img_rgb.shape[2], 1)
    
    return img_reshape

pickle_in1 = open('parameters.pickle', 'rb')
parameters = pickle.load(pickle_in1)

img_reshape = PreProcessImage(img)

def predict_HotDog_NotHotDog(img, parameters):
    
    Y_predict, _  = forward_propagation_n(img, parameters)
    
    if(Y_predict >= 0.5):
        return "Yay, it's a HotDog"
    else:
        return "Nay, it's not a HotDog"


