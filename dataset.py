from typing import Any, Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class PlanetsDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable]=None) -> None:
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self.data[idx]
    
    @property
    def classes(self) -> list[str]:
        return self.data.classes