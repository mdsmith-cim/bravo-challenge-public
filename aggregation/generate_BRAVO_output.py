import argparse
import os
from pathlib import Path
import numpy as np
import torch
from functools import partial
from tqdm.contrib.concurrent import process_map
from utils.score_functions import softmax
import cv2

parser = argparse.ArgumentParser(description='Ensemble evaluator')

parser.add_argument('folders', type=str, nargs='*', help='Folders containing the logit outputs of the various models we want to ensemble')
parser.add_argument('--out_path', type=str, help='Path to the folder to write BRAVO encode compatible output.', required=True)
parser.add_argument('--score_function', type=str, choices=('max_softmax_mean_softmax', 'rba_mean_logit', 'max_softmax_min_softmax'), default='max_softmax_mean_softmax',
                    help='Function to apply on logits to generate anomaly score at each pixel.'),
parser.add_argument('--label_function', type=str, choices=('max', 'mean', 'min'), default='mean', help='Function to apply on ensemble of logits to generate softmax vector (and subsequently the label via max) at each pixel.')

args = parser.parse_args()

logit_folders = []
for folder in args.folders:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder {folder} is not a directory!")
    logit_folders.append(folder)

print(f'Using {len(logit_folders)} models for ensemble evaluation')

out_path = Path(args.out_path)

def detect_dataset(logit_folders: list) -> str:
    possible_datasets = ('bravo_ACDC', 'bravo_SMIYC', 'bravo_synobjs', 'bravo_outofcontext','bravo_synflare', 'bravo_synrain')
    for f in logit_folders:
        for d in possible_datasets:
            if d in str(f):
                return d
detected_dataset = detect_dataset(logit_folders)
print(f'Detected dataset: {detected_dataset}')
out_path = out_path / detected_dataset
out_path.mkdir(exist_ok=True, parents=True)
assert out_path.is_dir(), f"Output path {out_path} does not exist!"

def loadProcessLogit(logit_data: dict, score_function: callable, label_function: callable, out_dir: Path):

    rel_path, logit_files = logit_data

    logit_array = []
    for logit_file in logit_files:
        logit = torch.as_tensor(np.load(logit_file)['logits'], dtype=torch.float32)
        logit_array.append(logit)
    logit_array = torch.stack(logit_array)
    softmaxes = []
    for l in logit_array:
        softmaxes.append(softmax(l))
    softmaxes = label_function(torch.stack(softmaxes))
    label = softmaxes.max(dim=0).indices
    anomaly_score = score_function(logit_array)

    base_out_name = rel_path.stem.replace('_logits', '')
    submit_fn = out_dir / rel_path.with_name(base_out_name + '_pred.png')
    submit_fn.parent.mkdir(exist_ok=True, parents=True)
    label_out = label.to(torch.uint8).numpy()
    write_success = cv2.imwrite(submit_fn, label_out)
    # Dump confidence in anomaly

    prob_out = (anomaly_score.numpy() * 65535).astype(np.uint16)
    submit_fn2 = out_dir / rel_path.with_name(base_out_name + '_conf.png')
    write_success &= cv2.imwrite(submit_fn2, prob_out)
    return write_success
def main():
    # Generate a lookup table of img file name -> paths to >=1 files with corresponding logits + GT mask
    logit_lookup = {}
    for logit_folder in logit_folders:
        logit_files = list(logit_folder.glob('**/*.npz'))
        for logit_file in logit_files:
            rel_path = logit_file.relative_to(logit_folder)
            if rel_path in logit_lookup:
                logit_lookup[rel_path].append(logit_file)
            else:
                logit_lookup[rel_path] = [logit_file]

    score_function = None
    if args.score_function == 'max_softmax_mean_softmax':
        from utils.score_functions_bravo import multisample_max_softmax_mean_softmax
        score_function = multisample_max_softmax_mean_softmax
    elif args.score_function == 'rba_mean_logit':
        from utils.score_functions_bravo import multisample_rba_mean_logit
        score_function = multisample_rba_mean_logit
    elif args.score_function == 'max_softmax_min_softmax':
        from utils.score_functions_bravo import multisample_max_softmax_min_softmax
        score_function = multisample_max_softmax_min_softmax

    label_function = None
    if args.label_function == 'max':
        from utils.label_functions import max as label_max
        label_function = label_max
    elif args.label_function == 'mean':
        from utils.label_functions import mean as label_mean
        label_function = label_mean
    elif not args.label_function == 'min':
        from utils.label_functions import min as label_min
        label_function = label_min

    results = process_map(partial(loadProcessLogit, score_function=score_function, label_function=label_function, out_dir=out_path), logit_lookup.items(), desc='Processing predictions...', unit='img', chunksize=2, max_workers=len(os.sched_getaffinity(0)))
    assert np.array(results).all(), "Failed to write all output files!"
    print(f'Wrote {len(results)} files to {out_path}')

if __name__ == '__main__':
    main()
