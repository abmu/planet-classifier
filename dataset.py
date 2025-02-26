from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL.Image import Image


class PlanetsDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable]=None) -> None:
        '''
        Args:
            data_dir: path to the dataset train, valid, or test folder
            transform: transform to be applied on data
        '''
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self) -> int:
        '''
        Returns the size of the dataset
        '''
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image, int]:
        '''
        Returns a tuple containing the image and its class label (integer in the range 0-9)
        '''
        return self.data[idx]
    
    @property
    def classes(self) -> list[str]:
        '''
        Returns the different classes in the dataset
        '''
        return self.data.classes


def get_dataloaders(train_dir: str, valid_dir: str, test_dir: str, batch_size: int=32) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Args:
        train_dir: path to the train folder
        valid_dir: path to the valid folder
        test_dir: path to the test folder
        batch_size: number of samples per batch, default 32

    Returns a tuple containing the data loaders for the train, valid, and test datasets
    '''

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])

    train_dataset = PlanetsDataset(data_dir=train_dir, transform=transform)
    valid_dataset = PlanetsDataset(data_dir=valid_dir, transform=transform)
    test_dataset = PlanetsDataset(data_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader