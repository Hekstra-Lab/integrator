from collections.abc import Callable

import torch
import torch.nn.functional as F

POSITIVE_CONSTRAINTS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": F.softplus,
    "exp": torch.exp,
}


def get_positive_constraint(
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if name not in POSITIVE_CONSTRAINTS:
        raise ValueError(
            f"positive_constraint must be one of {set(POSITIVE_CONSTRAINTS)}, got {name!r}"
        )
    return POSITIVE_CONSTRAINTS[name]
