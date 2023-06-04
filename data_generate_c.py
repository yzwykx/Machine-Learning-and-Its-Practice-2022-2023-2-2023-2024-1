import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle

def load_cifar10_batch(cifar10_dataset_folder_path,batch_id):

    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id),mode='rb') as file:
        batch = pickle.load(file, encoding = 'latin1')
        
    features = batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
    labels = batch['labels']
    
    return  features, labels 

cifar10_path = '/home/gengreen/ML/My_SVM/cifar-10-batches-py'

# train
x_train, y_train = load_cifar10_batch(cifar10_path, 1)

for i in range(2,6):
    features, labels = load_cifar10_batch(cifar10_path, i)
    x_train, y_train = np.concatenate([x_train, features]), np.concatenate([y_train, labels])


x_train = np.array(x_train).reshape(len(x_train),-1)
y_train = np.array(y_train)

labels0 = np.where(y_train == 0)
images0 = x_train[labels0]
labels1 = np.where(y_train == 1)
images1 = x_train[labels1]
labels2 = np.where(y_train == 2)
images2 = x_train[labels2]
labels3 = np.where(y_train == 3)
images3 = x_train[labels3]
labels4 = np.where(y_train == 4)
images4 = x_train[labels4]
labels5 = np.where(y_train == 5)
images5 = x_train[labels5]
labels6 = np.where(y_train == 6)
images6 = x_train[labels6]
labels7 = np.where(y_train == 7)
images7 = x_train[labels7]
labels8 = np.where(y_train == 8)
images8 = x_train[labels8]
labels9 = np.where(y_train == 9)
images9 = x_train[labels9]

imagesf = np.concatenate((images0[0:500],images1[0:500],images2[0:500],images3[0:500],images4[0:500],images5[0:500],images6[0:500],images7[0:500],images8[0:500],images9[0:500]),axis=0)
labelsf = np.concatenate((np.zeros(500),np.zeros(500)+1,np.zeros(500)+2,np.zeros(500)+3,np.zeros(500)+4,np.zeros(500)+5,np.zeros(500)+6,np.zeros(500)+7,np.zeros(500)+8,np.zeros(500)+9))

randomize = np.arange(5000)
np.random.shuffle(randomize)

imagesf = imagesf[randomize]
labelsf = labelsf[randomize]

np.save('images_c_train.npy',imagesf)
np.save('labels_c_train.npy',labelsf)

# test
x_test = None
y_test = None

with open(cifar10_path + '/test_batch', mode = 'rb') as file:
    batch = pickle.load(file, encoding = 'latin1')
    x_test = batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
    x_test = np.array(x_test).reshape(len(x_test),-1)
    y_test = batch['labels']
    y_test = np.array(y_test)

labels0 = np.where(y_test == 0)
images0 = x_test[labels0]
labels1 = np.where(y_test == 1)
images1 = x_test[labels1]
labels2 = np.where(y_test == 2)
images2 = x_test[labels2]
labels3 = np.where(y_test == 3)
images3 = x_test[labels3]
labels4 = np.where(y_test == 4)
images4 = x_test[labels4]
labels5 = np.where(y_test == 5)
images5 = x_test[labels5]
labels6 = np.where(y_test == 6)
images6 = x_test[labels6]
labels7 = np.where(y_test == 7)
images7 = x_test[labels7]
labels8 = np.where(y_test == 8)
images8 = x_test[labels8]
labels9 = np.where(y_test == 9)
images9 = x_test[labels9]

imagesf = np.concatenate((images0[0:100],images1[0:100],images2[0:100],images3[0:100],images4[0:100],images5[0:100],images6[0:100],images7[0:100],images8[0:100],images9[0:500]),axis=0)
labelsf = np.concatenate((np.zeros(100),np.zeros(100)+1,np.zeros(100)+2,np.zeros(100)+3,np.zeros(100)+4,np.zeros(100)+5,np.zeros(100)+6,np.zeros(100)+7,np.zeros(100)+8,np.zeros(100)+9))

randomize = np.arange(1000)
np.random.shuffle(randomize)

imagesf = imagesf[randomize]
labelsf = labelsf[randomize]

np.save('images_c_test.npy',imagesf)
np.save('labels_c_test.npy',labelsf)