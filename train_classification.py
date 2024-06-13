import os
import pandas as pd
from PIL import Image
import torch
import glob
import torchvision
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

class DogCatDataset(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.img_list = glob.glob(os.path.join(self.root,'train','*.jpg'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,index):

        img_path = self.img_list[index]
        label = os.path.basename(img_path).split('.')[0]

        if label =='cat':
            label = 1
        else:
            label = 0


        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img,label

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# train_dataset = DogCatDataset('D:/dataset/cat-and-dog/', transform=data_transforms['train'],mode = 'train')
# val_dataset = DogCatDataset('D:/dataset/cat-and-dog/', transform=data_transforms['val'],mode='val')
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
all_dataset = DogCatDataset('D:/dataset/cat-and-dog/', transform=data_transforms['train'])

train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size
train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,num_workers=2)
device = 'cuda'

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model
if __name__ == '__main__':
    
    model = train_model(model, criterion, optimizer, num_epochs=15)