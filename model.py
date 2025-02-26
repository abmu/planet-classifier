from torch import nn
import timm


class PlanetClassifier(nn.Module):
    def __init__(self, num_classes: int=10) -> None:
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)

    def forward(self, x):
        return output