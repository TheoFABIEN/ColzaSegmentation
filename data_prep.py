from torchvision import transforms 
from torch.utils.data import Dataset
import os
import skimage as ski
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%%


transform_images = transforms.Compose([
    transforms.Resize(size = 400, antialias = None),
    transforms.ToTensor()
])

transform_labels = transforms.Compose([
    #transforms.Grayscale(),
    transforms.Resize(size = 400, antialias = None),
    transforms.ToTensor()
])
#%%

class my_dataset(Dataset):

    def __init__(self, images_dir, mask_dir, transform_img = None, transform_lab = None):
        self.img_dir = images_dir
        self.mask_dir = mask_dir
        self.transform_images = transform_img
        self.transform_labels = transform_lab
        self.images = os.listdir(images_dir)
        self.labels = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.ROI.jpg', '.mac.png'))
        #image = ski.io.imread(img_path)
        #mask = ski.io.imread(mask_path)
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.transform_images:
            image = self.transform_images(image)
        if self.transform_labels:
            mask = self.transform_labels(mask)
        return image, mask
#%%

#test_data = my_dataset(
#        images_dir = '/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/ROI_avec GT',
#        mask_dir = '/home/theo/Bureau/Théo 2023 croiss lésions//Pour entrainer/GT', 
#        transform=transform
#)
#%%

