import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

# custom Dataset for road segmentation, separated into train/val/test folders
class RoadSegmentationDataset(Dataset):    
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        
        # paths to images and masks
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        
        # get all image filenames
        self.images = sorted(os.listdir(self.images_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # load mask (change extension from .jpg to .png)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_dataloaders(root_dir='./data', batch_size=4, augmentation=False, image_size=400):
    # Normalization values (may want to calculate these from our dataset)
    # reasonable defaults for satelite imagery
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # taken from previous assignmnet, if we use it some rotations would probably be plenty
    if augmentation:
        # Training transforms with augmentation
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Mask transforms for training (same geometric transforms, no color jitter)
        mask_transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ])
    else:
        # Training transforms without augmentation
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        mask_transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    # Test/Val transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    mask_transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    trainset = RoadSegmentationDataset(
        root_dir=root_dir,
        split='train',
        transform=transform_train,
        mask_transform=mask_transform_train
    )
    
    valset = RoadSegmentationDataset(
        root_dir=root_dir,
        split='val',
        transform=transform_test,
        mask_transform=mask_transform_test
    )
    
    testset = RoadSegmentationDataset(
        root_dir=root_dir,
        split='test',
        transform=transform_test,
        mask_transform=mask_transform_test
    )
    
    # Create dataloaders (pin_memory=False for MPS compatibility)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    return trainloader, valloader, testloader