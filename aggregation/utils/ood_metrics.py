import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
# Based on https://github.com/NazirNayal8/RbA/blob/main/support.py
def calculate_auroc(conf, gt):
    fpr, tpr, threshold = roc_curve(gt, conf)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0

    for i, j, k in zip(tpr, fpr, threshold):
        if i > 0.95:
            fpr_best = j
            break

    return roc_auc, fpr_best, k

def calculate_ood_metrics(out, label):
    prc_auc = average_precision_score(label, out)
    roc_auc, fpr, _ = calculate_auroc(out, label)
    return roc_auc, prc_auc, fpr

def calculate_out_and_label(anomaly_score, gt_mask):
    # (H, w) - indicating which image areas are in and out of distribution
    ood_mask = gt_mask == 1
    ind_mask = gt_mask == 0

    # Get the scores we gave for those in/out of dist areas
    # Note: 1-D area of scores, not image shape
    ood_out = anomaly_score[ood_mask]
    ind_out = anomaly_score[ind_mask]

    ood_label = torch.ones(len(ood_out))
    ind_label = torch.zeros(len(ind_out))

    val_out = torch.concatenate((ind_out, ood_out))
    val_label = torch.concatenate((ind_label, ood_label))
    return val_out, val_label

def calculate_ood(val_out, val_label, verbose=False):

    if verbose:
        print(f"Calculating Metrics for {len(val_out)} Points ...")

    auroc, aupr, fpr = calculate_ood_metrics(val_out, val_label)

    if verbose:
        print(f'AUROC score: {auroc}')
        print(f'AUPRC score: {aupr}')
        print(f'FPR@TPR95: {fpr}')

    return {'auroc': auroc, 'aupr': aupr, 'fpr95': fpr}