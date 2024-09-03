import torch
def max(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    return logit_array.max(dim=0).values

def mean(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    return logit_array.mean(dim=0)

def min(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    return logit_array.min(dim=0).values