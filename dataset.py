from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL.Image import Image


class PlanetsDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable]=None) -> None:
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image, int]:
        '''
        Returns a tuple containing the image and its class label (integer in the range 0-9)
        '''
        return self.data[idx]
    
    @property
    def classes(self) -> list[str]:
        return self.data.classes

dataset = PlanetsDataset(data_dir='./dataset/train')
print(len(dataset), dataset[0], dataset[1048])