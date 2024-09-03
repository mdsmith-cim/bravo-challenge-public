import numpy as np
import torch
from tabulate import tabulate
from .labels import trainId2label as cityscapesTrainId2label
from .labels import id2label as cityscapesid2label

def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

def calculate_semantic(label: torch.Tensor, gt_mask: torch.Tensor, dataset_name: str):
    assert label.shape == gt_mask.shape, f"Mask vs Label shape mismatch: {label.shape} != {gt_mask.shape}"

    if dataset_name == 'Cityscapes':

        trainID_gt_mask = torch.full_like(gt_mask, fill_value=255)
        for evalId, l in cityscapesid2label.items():
            trainID_gt_mask[gt_mask == evalId] = l.trainId
        gt_mask = trainID_gt_mask

    _iou_acc = fast_hist(label.numpy().flatten(),
                         gt_mask.numpy().flatten(),
                         19)
    return _iou_acc


def calculate_iou(hist_data):
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - \
        np.diag(hist_data)
    iu = np.diag(hist_data) / divisor
    return iu, acc, acc_cls

def print_evaluate_results(hist, iu, dataset_name):
    """
    If single scale:
       just print results for default scale
    else
       print all scale results

    Inputs:
    hist = histogram for default scale
    iu = IOU for default scale
    iou_per_scale = iou for all scales
    """
    if dataset_name == 'Cityscapes':
        id2cat = {trainId: l.name for trainId, l in cityscapesTrainId2label.items()}
    else:
        raise Exception(f"Dataset {dataset_name} not supported!")

    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)

    print('IoU:')

    header = ['Id', 'label']
    header.extend(['TP', 'FP', 'FN', 'Precision', 'Recall'])

    tabulate_data = []

    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)

        total_pixels = hist.sum()
        class_data.append(100 * iu_TP[class_id] / total_pixels)
        class_data.append(iu_FP[class_id] / iu_TP[class_id])
        class_data.append(iu_FN[class_id] / iu_TP[class_id])
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id]))
        class_data.append(iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id]))
        tabulate_data.append(class_data)

    print(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))
def eval_metrics(iou_acc, dataset_name, verbose=False):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory
    overflow for large dataset) Only applies to eval/eval.py
    """

    iu, acc, acc_cls = calculate_iou(iou_acc)

    if verbose:
        print_evaluate_results(iou_acc, iu, dataset_name)

    freq = iou_acc.sum(axis=1) / iou_acc.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    metrics = {
        'mean_iu': mean_iu,
        'acc_cls': acc_cls,
        'acc': acc,
        'fwavacc': fwavacc
    }
    if verbose:
        print('Mean: {:2.2f}'.format(mean_iu * 100))

    return metrics
