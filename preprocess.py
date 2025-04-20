import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Constants
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Standard transforms (resize, normalize, to tensor)
image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale mean/std based on ImageNet
])

# 2. Custom Dataset class to return patchified images
class PatchifiedXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=PATCH_SIZE):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.patch_size = patch_size
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img shape: [1, 224, 224] (if grayscale)
        if img.shape[0] == 3:
            img = transforms.Grayscale()(img)

        patches = self._patchify(img)
        return patches, label

    def _patchify(self, img):
        # img: [1, H, W] tensor
        img = img.squeeze(0)  # -> [H, W] for easier splitting
        patches = img.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        # shape: [num_patches_h, num_patches_w, patch_h, patch_w]
        patches = patches.contiguous().view(-1, self.patch_size * self.patch_size)
        return patches  # shape: [num_patches, patch_dim]

# 3. Loaders
def get_dataloaders(data_dir, batch_size=32):
    train_ds = PatchifiedXrayDataset(os.path.join(data_dir, 'train'), transform=image_transforms)
    val_ds = PatchifiedXrayDataset(os.path.join(data_dir, 'val'), transform=image_transforms)
    test_ds = PatchifiedXrayDataset(os.path.join(data_dir, 'test'), transform=image_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# In your training script:
train_loader, val_loader, test_loader = get_dataloaders('chest_xray')
for patch_batch, labels in train_loader:
    print(patch_batch.shape)  # [batch_size, num_patches, patch_dim]
    break

