import torch
from .score_functions import softmax, get_RbA
######
## Below is for single sample
######
def max_softmax(logits: torch.Tensor) -> torch.tensor:
    # Logits: (Classes, H, W)
    return softmax(logits).max(dim=0).values

######
## Below is for > 1 sample
######
def multisample_max_softmax_mean_softmax(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = torch.stack(softmaxes).mean(0)
    scores = softmaxes.max(dim=0).values
    return scores

def multisample_rba_mean_logit(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    logits = logit_array.mean(0)
    scores = get_RbA(logits)
    return scores

def multisample_max_softmax_min_softmax(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = torch.stack(softmaxes).min(dim=0).values
    scores = softmaxes.max(dim=0).values
    return scores