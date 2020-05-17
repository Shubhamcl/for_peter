from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

import numpy as np
from glob import glob
from os.path import join

data_path = "/home/shubham/Desktop/git/for_peter/test1/"


# Dataset

class CatsDogsDataset(Dataset):
    def __init__(self, data_address, transform, labels=None):

        self.paths = []
        self.transform = transform
        self.labels = []

        num_files = len(glob(join(data_address, '*')))
        for idx in range(1, num_files+1):
            self.paths.append(join(data_address, str(idx)+'.jpg'))
        
        if labels == None:
            self.labels = np.zeros((num_files, 3))
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = default_loader(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data loader making function

def make_data_loader(data_path, transforms, batch_size, num_workers, labels=None):
    ds = CatsDogsDataset(data_path, transforms, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader

if __name__ == "__main__":
    # Test
    from torchvision import transforms

    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    test_loader = make_data_loader(data_path, tfms, 16, 1)

    for batch in test_loader:
        print(batch[0])
        break   