import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class CNN(nn.Module):

    def __init__(self, file_json):
        super(CNN, self).__init__()
        with open(file_json, 'r') as f:
            params = json.load(f)

        self.frame_size = params['frame_Size']
        self.dropout = 0.2

        self.bn = nn.BatchNorm2d(num_features=8)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 7),
                stride=1,
            ),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(5, 7),
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(5, 7),
                stride=1,
            ),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*15*157, 513),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(513, 16),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Batch norm: N x M x F x T -> N x M x F x T
        x = self.bn(x)

        # Covolution layer1:
        x = self.layer1(x)

        # Covolution layer1:
        x = self.layer2(x)

        # Covolution layer1:
        x = self.layer3(x)

        # View: N x T x F x 2M > N x T x 2FM
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x