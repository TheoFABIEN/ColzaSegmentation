from U_net import * 
from data_prep import *
from torchvision import transforms
from PIL import Image
import glob 
import numpy as np 
import torch 
import os
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%%


img_path = '/home/theo/Bureau/Théo 2023 croiss lésions/2022 ROI faites'

#def segment_dataset(img_path):

files = glob.glob(img_path + '/*')
results_file = img_path + '_RESULTS'
results_argmax_file = img_path + '_RESULTS_SEUIL'

if os.path.exists(results_file):
    shutil.rmtree(results_file)
#%%

if os.path.exists(results_argmax_file):
    shutil.rmtree(results_argmax_file)
#%%
os.mkdir(results_file)
os.mkdir(results_argmax_file)

model = UNET(
        in_channels = 3,
        out_channels = 3
).to(DEVICE)

model.load_state_dict(
        torch.load(
            "/home/theo/Bureau/Théo 2023 croiss lésions/Code_segmentation/Trained_unets/trained_unet145.pt"
        )
)

#%%

for idx, imfile in enumerate(files):
    image = Image.open(imfile)
    image = transform_images(image)
    image = image.unsqueeze(0)
    print(image.shape)

    image = image.to(DEVICE)
    with torch.no_grad():
        result = model(image)
    result = result.cpu().numpy()[0,:,:,:].T
    ski.io.imsave(
        results_file + '/' + os.path.basename(imfile).replace('.jpg', '.tif').replace('.JPG', '.tif'),
        result
    )
    result_argmax = np.argmax(result, axis = 2)*100
    result_argmax = Image.fromarray(result_argmax.astype(np.uint8))
    result_argmax.save(
        results_argmax_file + 
        '/' + 
        os.path.basename(imfile).replace('.jpg', '.tif').replace('.JPG', '.tif')
    )

#%%

if __name__ == '__main__':
    img_path = '/home/theo/Bureau/Théo 2023 croiss lésions/sample'
    segment_dataset(img_path)

