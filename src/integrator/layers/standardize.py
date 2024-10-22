import torch


class Standardize(torch.nn.Module):
    def __init__(self, center=True, feature_dim=7, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.center = center

        # Register buffers for statistics
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("pixel_count", torch.tensor(0.0))

        # Add buffers for tracking update status
        self.register_buffer("total_pixels_seen", torch.tensor(0.0))
        self.register_buffer("should_update", torch.tensor(True))
        self.register_buffer("max_pixels", torch.tensor(float("inf")))

    def set_max_pixels(self, num_samples, pixels_per_sample):
        """Set the maximum number of pixels after which updates should stop"""
        self.max_pixels = torch.tensor(float(num_samples * pixels_per_sample))
        print(
            f"Standardization will stop updating after seeing {self.max_pixels} pixels"
        )

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / (self.pixel_count - 1).clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        # Don't update if we've exceeded max_pixels
        if not self.should_update:
            return

        # Input shape is [batch_size, pixels, features]
        B, P, F = im.shape

        if mask is None:
            mask = torch.ones((B, P), device=im.device)

        # Count valid pixels in current batch
        batch_count = mask.sum()

        if batch_count == 0:
            return

        # Update total pixels seen
        self.total_pixels_seen += batch_count

        # Check if we should stop updating
        if self.total_pixels_seen >= self.max_pixels:
            self.should_update = torch.tensor(False)
            print(
                f"\nStandardization updates stopped at {self.total_pixels_seen} pixels"
            )

        # Previous values
        old_pixel_count = self.pixel_count
        self.pixel_count += batch_count

        # Compute current batch mean (masked)
        masked_sum = (im * mask.unsqueeze(-1)).sum(dim=(0, 1))
        batch_mean = masked_sum / batch_count

        # Update mean using batch mean
        delta = batch_mean - self.mean.squeeze()
        self.mean = self.mean.squeeze() + delta * (batch_count / self.pixel_count)

        # Update M2 using the corrected formula
        if old_pixel_count > 0:
            # Update M2 using parallel algorithm
            delta_old = batch_mean - self.mean.squeeze()
            self.m2 += (
                (im - batch_mean.view(1, 1, -1))
                * mask.unsqueeze(-1)
                * (im - self.mean.squeeze().view(1, 1, -1))
            ).sum(dim=(0, 1))
            correction = (
                delta_old * delta * (old_pixel_count * batch_count / self.pixel_count)
            )
            self.m2 += correction
        else:
            # First batch
            self.m2 = (
                (im - self.mean.squeeze().view(1, 1, -1))
                * mask.unsqueeze(-1)
                * (im - self.mean.squeeze().view(1, 1, -1))
            ).sum(dim=(0, 1))

        # Reshape mean back to original shape
        self.mean = self.mean.view(1, 1, F)

    def standardize(self, im, mask=None):
        if not self.center:
            return im / self.std

        if mask is None:
            return (im - self.mean) / self.std

        return ((im - self.mean) * mask.unsqueeze(-1)) / self.std

    def forward(self, im, mask=None, training=True):
        if training and self.should_update:
            self.update(im, mask)
        return self.standardize(im, mask)
