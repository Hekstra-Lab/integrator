import torch
import torch.nn as nn


# class BackgroundIndicator(torch.nn.Module):
# def __init__(self, dmodel):
# super().__init__()
# # self.cnn = SimpleCNN()
# self.linear = Linear(dmodel, 3 * 21 * 21)

# def forward(self, x):
# x = self.cnn(x)
# bg_profile_params = self.linear(x)
# bg_profile = torch.sigmoid(bg_profile_params)
# return bg_profile


class BackgroundIndicator(torch.nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.fc = nn.Linear(
            dmodel, 4 * 4 * 64
        )  # First, increase the dimension to a small spatial grid

        self.conv_trans1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 8x8
        self.conv_trans2 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )  # 16x16
        self.conv_trans3 = nn.ConvTranspose2d(
            16, 3, kernel_size=6, stride=1, padding=0
        )  # 21x21

    def forward(self, x):
        # x is expected to have shape (batch_size, 1, dmodel)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, dmodel)

        x = self.fc(x)  # Apply the first linear transformation
        x = x.view(-1, 64, 4, 4)  # Reshape to start with a small spatial grid

        x = self.conv_trans1(x)  # Upsample to 8x8
        x = self.conv_trans2(x)  # Upsample to 16x16
        x = self.conv_trans3(x)  # Upsample to 21x21

        bg_profile = torch.sigmoid(
            x
        )  # Apply sigmoid to get the background profile in the range (0, 1)
        return bg_profile.view(bg_profile.size(0), 3 * 21 * 21)


# %%
