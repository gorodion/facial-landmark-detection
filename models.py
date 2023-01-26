from torch import nn

def convrelu(in_channels, out_channels, kernel, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
    )


KEYPOINT_DIM = 136


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp3x3 = nn.MaxPool2d(3, stride=2)
        self.mp2x2 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

        self.conv1 = convrelu(3, 32, kernel=3, padding=1)
        self.conv2 = convrelu(32, 64, kernel=3)
        self.conv3 = convrelu(64, 64, kernel=3)
        self.conv4 = convrelu(64, 128, kernel=2)

        self.projection = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
        self.head = nn.Linear(256, KEYPOINT_DIM)

    def forward(self, x):
        x = self.mp3x3(self.conv1(x))
        x = self.mp3x3(self.conv2(x))
        x = self.mp3x3(self.conv3(x))
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.projection(x)
        x = self.head(x)
        x = x.sigmoid()
        return x
