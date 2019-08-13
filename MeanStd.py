import cv2
import os
import numpy as np

path = os.path.join(os.getcwd(), 'voc2007', 'JPEGImages/')
mean_list = []
std_list = []
for i in os.listdir(path):
    image = cv2.imread(path+i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean, std = cv2.meanStdDev(image)
    mean_list.append(mean)
    std_list.append(std)

print(np.mean(np.array(mean_list), axis=0))
print(np.array(mean_list).shape)
print(np.mean(np.array(std_list), axis=0))
print(np.array(std_list).shape)
