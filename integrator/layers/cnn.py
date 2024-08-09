import torch
import torch.nn.functional as F


# # Define a simple CNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(
            10
        )  # Adjust to subregion size 10x10
        self.fc = torch.nn.Linear(128 * 10, 441 * 3)  # Output size for subregion 10x10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = x.view(x.size(0), 3, 21**2)  # Reshape to subregion 10x10
        return x
