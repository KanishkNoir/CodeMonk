import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

class FashionModel(nn.Module):
    def __init__(self, num_genders, num_article_types, num_base_colours, num_seasons):
        super(FashionModel, self).__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.gender_head = nn.Linear(num_features, num_genders)
        self.article_type_head = nn.Linear(num_features, num_article_types)
        self.season_head = nn.Linear(num_features, num_seasons)
        self.base_colour_head = nn.Linear(num_features, num_base_colours)
        self.base_season_head = nn.Linear(num_features, num_seasons)
        
    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        gender_out = self.gender_head(features)
        article_out = self.article_type_head(features)
        base_colour_out = self.base_colour_head(features)
        season_out = self.base_season_head(features)

        return gender_out, article_out, base_colour_out, season_out


