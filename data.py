from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_normalize():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

def get_dataset(data_folder, transfrom):
    return datasets.ImageFolder(
        root=data_folder, 
        transform=TwoCropTransform(transfrom)
    )

def set_loader(opt, size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.3, 0.1, 0.1)
        ], p=0.8),
        transforms.ToTensor(),
        get_normalize(),
    ])

    train_loader = DataLoader(
        get_dataset(opt.data_folder, train_transform),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=False,
        sampler=None,
        persistent_workers=True
    )

    return train_loader