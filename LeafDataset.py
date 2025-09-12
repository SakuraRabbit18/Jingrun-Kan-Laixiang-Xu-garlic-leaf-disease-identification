import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.data = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                label = self.class_to_idx[class_name]
                self.data.append((img_path, label))

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomDataset(root_dir='split_dataset/train', transform=transform)
    val_dataset = CustomDataset(root_dir='split_dataset/val', transform=transform)
    test_dataset = CustomDataset(root_dir='split_dataset/test', transform=transform)
    print(f"train_dataset len: {len(train_dataset)}")
    print(f"val_dataset len: {len(val_dataset)}")
    print(f"test_dataset len: {len(test_dataset)}")
    img, label = test_dataset[0]
    print(f"img.shape: {img.shape}")
    print(f"label: {label}")
    print(f"test_dataset.idx_to_class[label]: {test_dataset.idx_to_class[label]}")
