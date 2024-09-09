from torch import nn

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        n_channels = 1
        n_layers = 17
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(n_channels, features, kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, n_channels, kernel_size, padding=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)