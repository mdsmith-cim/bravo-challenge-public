import argparse
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from functools import partial
import concurrent.futures
from tqdm.auto import tqdm
from utils.ood_metrics import calculate_out_and_label
from utils.score_functions import softmax
from utils.semantic_metrics import calculate_semantic, eval_metrics

parser = argparse.ArgumentParser(description='Ensemble evaluator')

parser.add_argument('folders', type=str, nargs='*', help='Folders containing the logit outputs of the various models we want to ensemble')
parser.add_argument('--dataset_path', type=str, help='Path to the folder containing the chosen dataset.', required=True)
parser.add_argument('--dataset', type=str, choices=('SMIYC_anomaly', 'SMIYC_obstacle', 'Cityscapes'),
                    help='Dataset to use', required=True)
parser.add_argument('--score_function', type=str, choices=('max_softmax_mean_softmax','max_softmax_mean_logit','max_softmax_min_logit', 'max_softmax_min_softmax', 'test_func'), default='max_softmax_mean_softmax',
                    help='Function to apply on logits to generate anomaly score at each pixel.'),

args = parser.parse_args()
dataset_path = Path(args.dataset_path)

logit_folders = []
for folder in args.folders:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder {folder} is not a directory!")
    logit_folders.append(folder)

print(f'Using {len(logit_folders)} models for ensemble evaluation')
assert dataset_path.exists(), f"Dataset {dataset_path} does not exist!"

eval_type = 'ood'
if args.dataset == 'Cityscapes':
    eval_type = 'semantic'

def loadProcessLogit(logit_data: dict, score_function: callable):

    mask_path = logit_data['gt_mask']
    logit_files = logit_data['logit_files']
    gt_mask = torch.tensor(np.asarray(Image.open(mask_path)),
                           dtype=torch.int)  # as_tensor can't handle read-only NP array

    iou_acc, val_out, val_label = 0, None, None
    logit_array = []
    for logit_file in logit_files:
        # TODO: Explore alternatives to logit mean for anomaly scores
        logit = torch.as_tensor(np.load(logit_file)['logits'], dtype=torch.float32)
        logit_array.append(logit)
    logit_array = torch.stack(logit_array)

    if eval_type == 'semantic':
        softmaxes = []
        for l in logit_array:
            softmaxes.append(softmax(l))
        softmaxes = torch.stack(softmaxes).mean(0)
        label = softmaxes.max(dim=0).indices
        iou_acc = calculate_semantic(label, gt_mask, args.dataset)
    elif eval_type == 'ood':
        anomaly_score = score_function(logit_array)
        val_out, val_label = calculate_out_and_label(anomaly_score, gt_mask)
    return iou_acc, val_out, val_label

@staticmethod
def getMaskPath(dataset_path: Path, rel_path: Path, dataset_name: str) -> Path:
    if dataset_name in ('SMIYC_anomaly', 'SMIYC_obstacle'):
        base_filename = rel_path.stem.replace('_logits', '_labels_semantic.png')
        mask_path = dataset_path / 'labels_masks' / base_filename
        assert mask_path.exists(), f"Mask {mask_path} expected to exist!"
        return mask_path
    elif dataset_name == 'Cityscapes':
        new_name = rel_path.name.replace('_leftImg8bit_logits.npz', '_gtFine_labelIds.png')
        mask_path = dataset_path / 'gtFine_trainvaltest/gtFine' / rel_path.with_name(new_name)
        assert mask_path.exists(), f"Mask {mask_path} expected to exist!"
        return mask_path
    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")

def main():
    # Generate a lookup table of img file name -> paths to >=1 files with corresponding logits + GT mask
    logit_lookup = {}
    for logit_folder in logit_folders:
        logit_files = list(logit_folder.glob('**/*.npz'))
        for logit_file in logit_files:
            rel_path = logit_file.relative_to(logit_folder)
            if rel_path in logit_lookup:
                logit_lookup[rel_path]['logit_files'].append(logit_file)
            else:
                logit_lookup[rel_path] = {'logit_files': [logit_file], 'gt_mask': getMaskPath(dataset_path, rel_path, args.dataset)}

    score_function = None
    if args.score_function == 'max_softmax_mean_softmax':
        from aggregation.utils.score_functions import multisample_max_softmax_mean_softmax
        score_function = multisample_max_softmax_mean_softmax
    elif args.score_function == 'max_softmax_mean_logit':
        from aggregation.utils.score_functions import multisample_max_softmax_mean_logit
        score_function = multisample_max_softmax_mean_logit
    elif args.score_function == 'max_softmax_min_logit':
        from aggregation.utils.score_functions import multisample_max_softmax_min_logit
        score_function = multisample_max_softmax_min_logit
    elif args.score_function == 'test_func':
        from aggregation.utils.score_functions import test_func
        score_function = test_func
    elif args.score_function == 'max_softmax_min_softmax':
        from aggregation.utils.score_functions import multisample_max_softmax_min_softmax
        score_function = multisample_max_softmax_min_softmax

    iou_acc = 0
    val_label_all, val_out_all = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as executor:
        results = executor.map(partial(loadProcessLogit, score_function=score_function), logit_lookup.values(), chunksize=2)
        for r in tqdm(results, total=len(logit_lookup), desc='Processing predictions...', unit='img'):
            if eval_type == 'semantic':
                iou_acc += r[0]
            elif eval_type == 'ood':
                val_out_all.append(r[1])
                val_label_all.append(r[2])


    if eval_type == 'ood':
        from aggregation.utils.ood_metrics import calculate_ood
        val_out_all = torch.concatenate(val_out_all)
        val_label_all = torch.concatenate(val_label_all)
        results = calculate_ood(val_out_all, val_label_all)
        print(f'For dataset {args.dataset} with score function {args.score_function}:')
        print(f'AUROC score: {results["auroc"]}')
        print(f'AUPRC score: {results["aupr"]}')
        print(f'FPR@TPR95: {results["fpr95"]}')
    elif eval_type == 'semantic':
        print(f'For dataset {args.dataset} semantic scores:')
        results = eval_metrics(iou_acc, args.dataset, True)


if __name__ == '__main__':
    main()
