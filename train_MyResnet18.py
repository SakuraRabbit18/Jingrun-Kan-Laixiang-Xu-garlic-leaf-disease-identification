import time
from PLTDataset import LicensePlateDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyResBet18 import ModifiedResNet18
from Triplet import TripletAttention
import torch.nn as nn
from torch.optim import Adam
import train_function
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from LeafDataset import CustomDataset


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = CustomDataset(root_dir='split_dataset/train', transform=transform)
    val_dataset = CustomDataset(root_dir='split_dataset/val', transform=transform)
    test_dataset = CustomDataset(root_dir='split_dataset/test', transform=transform)

    epoch = 100

    custom_batch_size=4
    works = 4
    prefetch=2
    train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,prefetch_factor=prefetch,num_workers=works)
    val_loader = DataLoader(val_dataset, batch_size=custom_batch_size, shuffle=True,prefetch_factor=prefetch,num_workers=works)
    test_loader = DataLoader(test_dataset, batch_size=custom_batch_size, shuffle=True,prefetch_factor=prefetch,num_workers=works)


    print("cuda:",torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_div = 2 
    model = ModifiedResNet18(n_div, 5).to(device)
    custom_module = TripletAttention()
    model.resnet.conv1 = nn.Sequential(
        model.resnet.conv1,
        custom_module
    ).to(device)

    file_name = 'myResNet18_bs4_ep10_224_varlr'
    start = time.time()
    acc,loss,_,_ = train_function.train_var_lr(model=model,train_loader=train_loader,num_epochs=epoch,device=device,save_name=file_name,val_loader=val_loader)
    end = time.time()
    total_time = end - start
    print(f"Total time：{total_time}")
    train_function.test_model(model=model,test_loader=test_loader,device=device)
