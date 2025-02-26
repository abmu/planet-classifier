from torch import nn, Tensor
import timm


class PlanetClassifier(nn.Module):
    def __init__(self, num_classes: int=10) -> None:
        '''
        Args:
            num_classes: the number of classes in the output
        '''
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # Create copy of the base model without the last layer (classifier)
        self.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: input tensor of shape (batch_size, channels, height, width)

        Returns output tensor of shape (batch_size, num_classes)
        '''
        x = self.features(x)
        output = self.classifier(x)
        return output