import torch.nn.functional as F
import torch
######
## Below is for single sample
######
def softmax(logits: torch.Tensor) -> torch.Tensor:
    # Logits: (Classes, H, W)
    return F.softmax(logits, dim=0)
def max_softmax(logits: torch.Tensor) -> torch.tensor:
    # Logits: (Classes, H, W)
    return 1 - softmax(logits).max(dim=0).values

def _entropy(pk: torch.Tensor, dim=0) -> torch.Tensor:
    """
    Adapted from scipy https://github.com/scipy/scipy/blob/v1.9.1/scipy/stats/_entropy.py#L17-L88
    Simplified for my uses, and also adapted to use function calls on tensors for CUDA memory efficiency in PyTorch.
    This routine will normalize `pk` if it doesn't sum to 1.
    Entropy calculated as ``S = -sum(pk * log(pk), axis=axis)``.
    :param pk: Tensor
        Defines the (discrete) distribution. Along each axis-slice of ``pk``,
        element ``i`` is the  (possibly unnormalized) probability of event
        ``i``.
    :param dim: int, optional
        The axis along which the entropy is calculated. Default is 0.
    :return: S : {float, Tensor}
        The calculated entropy.
    """
    pk = 1.0 * pk / torch.sum(pk, dim=dim, keepdim=True)
    return torch.sum(-pk * pk.log(), dim=dim)

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Logits: (Classes, H, W)
    return _entropy(softmax(logits), dim=0)

def get_RbA(logits: torch.Tensor) -> torch.Tensor:
    # Logits: (Classes, H, W)
    # From https://github.com/NazirNayal8/RbA/tree/main
    return -logits.tanh().sum(dim=0)

def get_energy(logits: torch.Tensor) -> torch.Tensor:
    # Logits: (Classes, H, W)
    return -torch.logsumexp(logits, dim=0)

######
## Below is for > 1 sample
######
def multisample_max_softmax_mean_softmax(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = torch.stack(softmaxes).mean(0)
    scores = 1 - softmaxes.max(dim=0).values
    return scores

def multisample_max_softmax_mean_logit(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    logits = logit_array.mean(0)
    scores = 1 - softmax(logits).max(dim=0).values
    return scores

def multisample_max_softmax_min_logit(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = torch.stack(softmaxes).min(0).values
    scores = 1 - softmaxes.max(dim=0).values
    return scores

def multisample_max_softmax_min_softmax(logit_array: torch.Tensor) -> torch.Tensor:
    # all_logits: (Samples, Classes, H, W)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = torch.stack(softmaxes).min(dim=0).values
    scores = 1 - softmaxes.max(dim=0).values
    return scores

def multisample_min_max_softmax_per_sample(logit_array: torch.Tensor) -> torch.Tensor:
    #multisample_max_max_softmax_per_sample
    # all_logits: (Samples, Classes, H, W)
    max_softmaxes = []
    for l in logit_array:
        max_softmaxes.append(softmax(l).max(0).values)
    max_softmaxes = torch.stack(max_softmaxes)
    scores = 1 - max_softmaxes.min(dim=0).values
    return scores

def test_func(logit_array: torch.Tensor) -> torch.Tensor:
    # entropy of mean
    # all_logits: (Samples, Classes, H, W)
    # softmaxes = []
    # for l in logit_array:
    #     softmaxes.append(softmax(l))
    # softmaxes = torch.stack(softmaxes).mean(0)
    # scores = _entropy(softmaxes, dim=0)

    # mean -> max logit
    # all_logits: (Samples, Classes, H, W)
    #scores = -logit_array.mean(0).max(dim=0).values

    # min get minner, max get maxxer
    # all_logits: (Samples, Classes, H, W)
    # best_value = logit_array[0].detach().clone()
    # for i in range(1, len(logit_array)):
    #     cur_value = logit_array[i]
    #     min_match = (cur_value < 0) & (cur_value < best_value)
    #     max_match = (cur_value >= 0) & (cur_value > best_value)
    #     best_value[min_match] = cur_value[min_match]
    #     best_value[max_match] = cur_value[max_match]
    #
    # scores = 1 - softmax(best_value).max(dim=0).values

    # Forcibly assign low confidence
    # softmaxes = []
    # for l in logit_array:
    #     softmaxes.append(softmax(l))
    # softmaxes = torch.stack(softmaxes).mean(0)
    # scores = softmaxes.max(dim=0).values
    # threshold = 0.4
    # scores[scores >= threshold] = 1
    # scores[scores < threshold] = 0
    # scores = 1 - scores

    # Sample entropy
    # all_logits: (Samples, Classes, H, W)
    # softmaxes = []
    # for l in logit_array:
    #     softmaxes.append(softmax(l))
    # softmaxes = torch.stack(softmaxes)
    # scores = _entropy(softmaxes, dim=0).mean(0)

    # Mean -> RbA
    # all_logits: (Samples, Classes, H, W)
    scores = get_RbA(logit_array[1])
    return scores