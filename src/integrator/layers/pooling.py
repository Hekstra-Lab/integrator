import torch


class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        data = data * mask
        out = torch.sum(data, dim=1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-2, keepdim=True)
        out = out / denom

        return out.squeeze(1)
