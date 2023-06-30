from U_net import * 
from data_prep import transform_images, transform_labels, my_dataset 
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader, random_split
import torch 
import skimage as ski
import seaborn as sns 
import matplotlib.pyplot as plt

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

#%% Data importation

train_img_dir = '/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/ROI_avec GT' 
train_masks_dir = '/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/GT' 
test_img_dir = '/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/test_set/test_images' 
test_masks_dir = '/home/theo/Bureau/Théo 2023 croiss lésions/Pour entrainer/test_set/test_labels' 

train_set = my_dataset(
    train_img_dir, 
    train_masks_dir, 
    transform_img = transform_images,
    transform_lab = transform_labels
)
test_set = my_dataset(
    test_img_dir, 
    test_masks_dir, 
    transform_img = transform_images,
    transform_lab = transform_labels
)

train, valid = random_split(train_set, [.8, .2])
train_loader = DataLoader(train, batch_size = 4, shuffle = True, pin_memory = True)
valid_loader = DataLoader(valid, batch_size = 4, shuffle = True, pin_memory = True)


#%%

model = UNET(in_channels = 3, out_channels = 3).to(DEVICE)

#%%

NUM_EPOCH = 20

#loss_fn = JaccardIndex(task = 'multiclass', num_classes = 3).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 1e-2)

train_losses = []
valid_losses = []
#%%

for epoch in range(NUM_EPOCH):

    totaltrainloss = 0
    totalvalloss = 0

    for i, data in enumerate(train_loader):

        opt.zero_grad()
        inputs, labels = data 
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        labels = labels[:,0,:,:] 
        labels = labels*255
        labels = labels.type(torch.long)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        totaltrainloss += loss

    train_losses.append(totaltrainloss.item())

    with torch.no_grad():
        for idx, vdata in enumerate(valid_loader):
            vinputs, vlabels = vdata 
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
            vlabels = vlabels[:,0,:,:] 
            vlabels = vlabels*255
            vlabels = vlabels.type(torch.long)
            voutputs = model(vinputs)
            val_loss = loss_fn(voutputs, vlabels)
            totalvalloss += val_loss

    valid_losses.append(totalvalloss.item())

    print(f"Epoch : {epoch}, Train loss : {totaltrainloss}, Validation loss : {totalvalloss}")

    if i%5 == 0:
        torch.save(
            model.state_dict(),
            "/home/theo/Bureau/Théo 2023 croiss lésions/Code_segmentation/trained_unet" + str(i) + ".pt"
        )


#%%  Plot results

sns.lineplot(
        x = range(NUM_EPOCH),
        y = train_losses
)

sns.lineplot(
        x = range(NUM_EPOCH),
        y = valid_losses
), plt.show()


#%%   Save model

#torch.save(model.state_dict(), "/home/theo/Bureau/Théo 2023 croiss lésions/Code_segmentation/trained_unet.pt")
