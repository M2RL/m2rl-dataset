import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

class RGBFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, film_input_size=10):
        super(RGBFeatureExtractor, self).__init__()
        self.backbone = ResNetBackboneWithFiLM(pretrained=pretrained, film_input_size=film_input_size).to('cuda')
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x, film_input=None):
        features = self.backbone(x, film_input)
        return features

class FiLMLayer(nn.Module):
    def __init__(self, input_size, num_features):
        super(FiLMLayer, self).__init__()
        self.gamma = nn.Linear(input_size, num_features)
        self.beta = nn.Linear(input_size, num_features)
        
    def forward(self, x, film_input):
        gamma = self.gamma(film_input).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(film_input).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ResNetBackboneWithFiLM(nn.Module):
    def __init__(self, pretrained=True, film_input_size=10):
        super(ResNetBackboneWithFiLM, self).__init__()
        backbone = resnet18(pretrained=pretrained)
        
        # Remove the fully connected layer
        layers = list(backbone.children())[:-1]
        
        # Add FiLM layers after each residual block
        self.film_layers = nn.ModuleList()
        for layer in layers[4:]:
            if isinstance(layer, nn.Sequential):
                blocks = list(layer.children())
                for block in blocks:
                    if isinstance(block, BasicBlock):
                        self.film_layers.append(FiLMLayer(film_input_size, block.conv2.out_channels))
        self.backbone = nn.Sequential(*layers)
        
    def forward(self, x, film_input=None):
        features = x
        film_layer_idx = 0
        for layer in self.backbone:
            features = layer(features)
            if isinstance(layer, nn.Sequential):
                blocks = list(layer.children())
                for block in blocks:
                    if isinstance(block, BasicBlock):
                        if film_input is not None:
                            features = self.film_layers[film_layer_idx](features, film_input)
                        film_layer_idx += 1
        return features

if __name__ == "__main__":
    # Create an example RGB image and FiLM input
    example_image = torch.rand(1, 3, 224, 224)
    example_film_input = torch.rand(1, 10)
    
    # Create the feature extractor
    feature_extractor = RGBFeatureExtractor(pretrained=True, freeze_backbone=False, film_input_size=10)
    
    # Extract features with FiLM modulation
    features = feature_extractor(example_image, example_film_input)
    print(features.shape)        