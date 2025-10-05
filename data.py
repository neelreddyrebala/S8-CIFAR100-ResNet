import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def build_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tfms, test_tfms

def get_loaders(data_dir='./data', batch_size=128, workers=4):
    train_tfms, test_tfms = build_transforms()

    train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tfms)
    test_ds  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader
