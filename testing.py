from data_prep import * 
from U_net import * 
import matplotlib.pyplot as plt 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

 #%%

test_dataset = my_dataset(
        "/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/test_set/test_images", 
        "/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/test_set/test_labels/",
        transform_img = transform_images,
        transform_lab = transform_labels
)


test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)


model = UNET(
        in_channels = 3,
        out_channels = 3
).to(DEVICE)

model.load_state_dict(
        torch.load("/home/theo/Bureau/Théo 2023 croiss lésions/Code_segmentation/Trained_unets/trained_unet145.pt")
)


#%%

jac_scores = []

with torch.no_grad():

    for idx, data in enumerate(test_loader):
    
        img, label = data 
        img = img.to(DEVICE)
        pred = model(img)
        pred = pred.cpu().numpy().T 
        label = label[0].numpy().T*255
        print(f'Image{idx} done')
        plt.subplot(131)
        plt.imshow(pred[:,:,:,0])
        plt.subplot(132)
        plt.imshow(np.argmax(pred[:,:,:,0], axis = 2))
        plt.subplot(133)
        plt.imshow(label), plt.show()

        jac_scores.append(
                jaccard_score(
                    pd.DataFrame(np.argmax(pred[:,:,:,0], axis = 2).flatten()),
                    pd.DataFrame(label.flatten()),
                    average = 'macro'
                )
        )

#%%

mean_jacc_score = np.mean(jac_scores) 
print(f"Mean Jaccard score = {mean_jacc_score}")
