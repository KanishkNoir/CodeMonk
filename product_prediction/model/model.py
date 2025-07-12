import torch.nn as nn
from torchvision import models

class FashionModel(nn.Module):
    def __init__(self, num_genders, num_article_types, num_base_colours, num_seasons):
        super(FashionModel, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        num_features = backbone.fc.in_features

        self.gender_head = nn.Linear(num_features, num_genders)
        self.article_type_head = nn.Linear(num_features, num_article_types)
        self.base_colour_head = nn.Linear(num_features, num_base_colours)
        self.season_head = nn.Linear(num_features, num_seasons)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        gender_out = self.gender_head(features)
        article_out = self.article_type_head(features)
        base_colour_out = self.base_colour_head(features)
        season_out = self.season_head(features)
        return gender_out, article_out, base_colour_out, season_out