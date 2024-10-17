import torch


class Standardize(torch.nn.Module):
    def __init__(
        self, center=True, feature_dim=7, max_counts=float("inf"), epsilon=1e-6
    ):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("pixel_count", torch.tensor(0.0))  # Counter for pixels
        self.register_buffer("image_count", torch.tensor(0.0))  # Counter for images

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / self.pixel_count.clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        if mask is None:
            k = len(
                im
            )  # Assuming 'k' should be number of elements in 'im' if no mask is provided
        else:
            k = mask.sum()  # count num of pixels in batch
        self.pixel_count += k
        self.image_count += len(im)

        if mask is None:
            diff = im - self.mean
        else:
            diff = (im - self.mean) * mask.unsqueeze(-1)

        new_mean = self.mean + torch.sum(diff, dim=(0, 1)) / self.pixel_count
        if mask is None:
            self.m2 += torch.sum((im - new_mean) * diff, dim=(0, 1))
        else:
            self.m2 += torch.sum(
                (im - new_mean) * mask.unsqueeze(-1) * diff, dim=(0, 1)
            )
        self.mean = new_mean

    def standardize(self, im, mask=None):
        if self.center:
            if mask is None:
                return (im - self.mean) / self.std
            else:
                return ((im - self.mean) * mask.unsqueeze(-1)) / self.std
        return im / self.std

    def forward(self, im, mask=None, training=True):
        if self.image_count >= self.max_counts:
            training = False

        if training:
            self.update(im, mask)

        return self.standardize(im, mask)
