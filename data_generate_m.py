import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import cv2
import matplotlib.pyplot as plt

class MNIST(Dataset):

    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)
  
def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)

trainDataset = MNIST('./mnist_data', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
testDataset = MNIST('./mnist_data', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=1,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=1,
    shuffle=False,
)

# train
images = []
labels = []

for img, lab in iter(train_loader):
    image = img.tolist()
    label = lab.tolist()
    images.append(image)
    labels.append(label)
    
images = np.array(images).reshape(len(images),-1)
labels = np.array(labels)

labels0, _ = np.where(labels == 0)
images0 = images[labels0]
labels1, _ = np.where(labels == 1)
images1 = images[labels1]
labels2, _ = np.where(labels == 2)
images2 = images[labels2]
labels3, _ = np.where(labels == 3)
images3 = images[labels3]
labels4, _ = np.where(labels == 4)
images4 = images[labels4]
labels5, _ = np.where(labels == 5)
images5 = images[labels5]
labels6, _ = np.where(labels == 6)
images6 = images[labels6]
labels7, _ = np.where(labels == 7)
images7 = images[labels7]
labels8, _ = np.where(labels == 8)
images8 = images[labels8]
labels9, _ = np.where(labels == 9)
images9 = images[labels9]

imagesf = np.concatenate((images0[0:500],images1[0:500],images2[0:500],images3[0:500],images4[0:500],images5[0:500],images6[0:500],images7[0:500],images8[0:500],images9[0:500]),axis=0)
labelsf = np.concatenate((np.zeros(500),np.zeros(500)+1,np.zeros(500)+2,np.zeros(500)+3,np.zeros(500)+4,np.zeros(500)+5,np.zeros(500)+6,np.zeros(500)+7,np.zeros(500)+8,np.zeros(500)+9))

randomize = np.arange(5000)
np.random.shuffle(randomize)

imagesf = imagesf[randomize]
labelsf = labelsf[randomize]

np.save('images_m_train.npy',imagesf)
np.save('labels_m_train.npy',labelsf)

# test
images = []
labels = []

for img, lab in iter(test_loader):
    image = img.tolist()
    label = lab.tolist()
    images.append(image)
    labels.append(label)
    
images = np.array(images).reshape(len(images),-1)
labels = np.array(labels)

labels0, _ = np.where(labels == 0)
images0 = images[labels0]
labels1, _ = np.where(labels == 1)
images1 = images[labels1]
labels2, _ = np.where(labels == 2)
images2 = images[labels2]
labels3, _ = np.where(labels == 3)
images3 = images[labels3]
labels4, _ = np.where(labels == 4)
images4 = images[labels4]
labels5, _ = np.where(labels == 5)
images5 = images[labels5]
labels6, _ = np.where(labels == 6)
images6 = images[labels6]
labels7, _ = np.where(labels == 7)
images7 = images[labels7]
labels8, _ = np.where(labels == 8)
images8 = images[labels8]
labels9, _ = np.where(labels == 9)
images9 = images[labels9]

imagesf = np.concatenate((images0[0:100],images1[0:100],images2[0:100],images3[0:100],images4[0:100],images5[0:100],images6[0:100],images7[0:100],images8[0:100],images9[0:500]),axis=0)
labelsf = np.concatenate((np.zeros(100),np.zeros(100)+1,np.zeros(100)+2,np.zeros(100)+3,np.zeros(100)+4,np.zeros(100)+5,np.zeros(100)+6,np.zeros(100)+7,np.zeros(100)+8,np.zeros(100)+9))

randomize = np.arange(1000)
np.random.shuffle(randomize)

imagesf = imagesf[randomize]
labelsf = labelsf[randomize]

np.save('images_m_test.npy',imagesf)
np.save('labels_m_test.npy',labelsf)