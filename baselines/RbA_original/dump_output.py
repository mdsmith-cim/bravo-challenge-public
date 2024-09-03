import os
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_net import Trainer, setup
from detectron2.checkpoint import DetectionCheckpointer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.cityscapes import Cityscapes
from datasets.segment_me_if_you_can import RoadAnomaly21, RoadObstacle21
from datasets.bravo import BRAVO
from easydict import EasyDict as edict
import cv2

AVAIL_DATASETS = ['cityscapes', 'road_anomaly_21', 'road_obstacles', 'bravo_ACDC', 'bravo_SMIYC', 'bravo_outofcontext', 'bravo_synflare', 'bravo_synobjs', 'bravo_synrain']

def get_test_datasets(datasets_folder, dataset_mode, selected_datasets):

    assert dataset_mode in ('selective', 'all')
    cityscapes_config = edict(
        dataset_root=os.path.join(datasets_folder, 'Cityscapes-nvidia'),
        return_filepath=True
    )

    road_anomaly_21_config = edict(
        dataset_root=os.path.join(datasets_folder,
                                'SegmentMeIfYouCan'),
        dataset_mode='val',
        return_filepath=True
    )

    road_obstacle_21_config = edict(
        dataset_root=os.path.join(datasets_folder,
                                'SegmentMeIfYouCan'),
        dataset_mode='val',
        return_filepath=True
    )

    bravo_ACDC_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_ACDC',
        return_filepath=True
    )
    bravo_SMIYC_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_SMIYC',
        return_filepath=True
    )
    bravo_outofcontext_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_outofcontext',
        return_filepath=True
    )
    bravo_synflare_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synflare',
        return_filepath=True
    )
    bravo_synobjs_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synobjs',
        return_filepath=True
    )
    bravo_synrain_config = edict(
        dataset_root=os.path.join(datasets_folder),
        dataset_mode='bravo_synrain',
        return_filepath=True
    )

    transform = A.Compose([
        ToTensorV2()
    ])

    if dataset_mode == 'all':
        selected_datasets = AVAIL_DATASETS

    DATASETS = edict()
    if 'cityscapes' in selected_datasets:
        DATASETS.cityscapes = Cityscapes(cityscapes_config, transform=transform, split='val', target_type='semantic')
    if 'road_anomaly_21' in selected_datasets:
        DATASETS.road_anomaly_21 = RoadAnomaly21(hparams=road_anomaly_21_config, transforms=transform)
    if 'road_obstacles' in selected_datasets:
        DATASETS.road_obstacles = RoadObstacle21(road_obstacle_21_config, transforms=transform)
    if 'bravo_ACDC' in selected_datasets:
        DATASETS.bravo_ACDC = BRAVO(hparams=bravo_ACDC_config, transforms=transform)
    if 'bravo_SMIYC' in selected_datasets:
        DATASETS.bravo_SMIYC = BRAVO(hparams=bravo_SMIYC_config, transforms=transform)
    if 'bravo_outofcontext' in selected_datasets:
        DATASETS.bravo_outofcontext = BRAVO(hparams=bravo_outofcontext_config, transforms=transform)
    if 'bravo_synflare' in selected_datasets:
        DATASETS.bravo_synflare = BRAVO(hparams=bravo_synflare_config, transforms=transform)
    if 'bravo_synobjs' in selected_datasets:
        DATASETS.bravo_synobjs = BRAVO(hparams=bravo_synobjs_config, transforms=transform)
    if 'bravo_synrain' in selected_datasets:
        DATASETS.bravo_synrain = BRAVO(hparams=bravo_synrain_config, transforms=transform)

    return DATASETS

parser = argparse.ArgumentParser(description='OOD Test Set Predictor')

parser.add_argument('--batch_size', type=int, default=1,
                    help="Batch Size used in evaluation")
parser.add_argument('--num_workers', type=int, default=15,
                    help="Number of threads used in data loader")
parser.add_argument('--device', type=str, default='cuda',
                    help="cpu or cuda, the device used for evaluation")
parser.add_argument('--out_path', type=str, default='results',
                    help='Output directory to save results to')
parser.add_argument('--verbose', type=bool, default=True,
                    help="If True, the records will be printed every time they are saved")
parser.add_argument('--datasets_folder', type=str, default='datasets',
                    help='the path to the folder that contains all datasets for evaluation')
parser.add_argument('--models_folder', type=str, default='ckpts/',
                    help='the path that contains the models to be evaluated')
parser.add_argument('--model_mode', type=str, default='all', choices=['all', 'selective'],
                    help="""One of [all, selective]. Defines which models to evaluate, the default behavior is all, which is to 
                            evaluate all models in model_logs dir. You can also choose particular models
                            for evaluation, in which case you need to pass the names of the models to --selected_models""")
parser.add_argument("--selected_models", nargs="*", type=str, default=[],
                    help="Names of models to be evaluated, these should be name of directories in model_logs")
parser.add_argument('--dataset_mode', type=str, default='all', choices=['all', 'selective'],
                    help="""One of [all, selective]. Defines which datasets to evaluate on, the default behavior is all, which is to 
                            evaluate all available datasets. You can also choose particular datasets
                            for evaluation, in which case you need to pass the names of the datasets to --selected_datasets.""")
parser.add_argument("--selected_datasets", nargs="*", type=str, default=[],
                    choices=AVAIL_DATASETS,
                    help="""Names of datasets to be evaluated.
                    """)
parser.add_argument('--dump_logits', action='store_true', help='If set, dump raw logit outputs rather than 8-bit PNG predictions.')
parser.add_argument("--score_func", type=str, default="rba", choices=["rba", "pebal"],
                    help="Outlier scoring function to be used in evaluations. Note: dense_hybrid not supported at the moment, and not applicable if --dump_logits is set.")

args = parser.parse_args()

if 'bravo_SMIYC' in args.selected_datasets:
    assert args.batch_size == 1, "BRAVO SMIYC dataset only supports batch size of 1 due to image size differences"

DATASETS = get_test_datasets(args.datasets_folder, args.dataset_mode, args.selected_datasets)

dataset_group = [(name, dataset) for (name, dataset) in DATASETS.items()]

print("Datasets to be evaluated:")
[print(g[0]) for g in dataset_group]
print("-----------------------")

# Device for computation
if args.device == 'cuda' and (not torch.cuda.is_available()):
    print("Warning: Cuda is requested but cuda is not available. CPU will be used.")
    args.device = 'cpu'
DEVICE = torch.device(args.device)


def get_model(config_path, model_path):
    """
    Creates a Mask2Former model give a config path and ckpt path
    """
    args = edict({'config_file': config_path, 'eval-only': True, 'opts': [
        "OUTPUT_DIR", "output/",
    ]})
    config = setup(args)

    model = Trainer.build_model(config)
    DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        model_path, resume=False
    )
    model.to(DEVICE)
    _ = model.eval()

    return model


def get_RbA(logits):
    """
    Expected input (batch size, 19, H, W)
    """
    return -logits.tanh().sum(dim=1)

def get_energy(logits):
    """
    Expected input (batch size, 19, H, W)
    """
    return -torch.logsumexp(logits, dim=1)


# This seems to require a model that actually allows return_ood_pred
# For now just don't bother
# def get_densehybrid_score(model, x):
#     with torch.no_grad():
#         out, ood_pred = model([{"image": x[0].cuda()}], return_ood_pred=True)
#
#     logits = out[0]['sem_seg']
#
#     out = F.softmax(ood_pred, dim=1)
#     p1 = torch.logsumexp(logits, dim=0)
#     p2 = out[:, 1]  # p(~din|x)
#     probs = (- p1) + (p2 + 1e-9).log()
#     conf_probs = probs
#     return conf_probs
def get_model_output(model, score_func, x):
    """
    Returns the output of the model given an input x and a score function.
    Expected input:
    - model: torch model
    - score_func: function that takes logits as input and produces pixel-wise anomaly score e.g. RbA
    - x: torch.Tensor of shape (batch size, 3, H, W)
    Expected output:
    - Logits (torch.Tensor) of shape (batch size, 19, H, W)
    - Score: (torch.tensor) of shape (batch size, H, W)
    @return: (model output score, logits)
    """
    with torch.no_grad():
        #out = model([{"image": x[0]}])
        out = model([{"image": img} for img in x])

    logits = torch.stack([o['sem_seg'] for o in out], dim=0)

    return score_func(logits), logits


def generate_predictions(model, dataset, dataset_name, model_name, out_path):
    """
    Generate predictions for a particular model over all designated datasets, saving results to appropriate folders.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    score_func = None
    if args.score_func == "rba":
        score_func = get_RbA
    elif args.score_func == "pebal":
        score_func = get_energy
    # elif args.score_func == "dense_hybrid":
    #     score_func = get_densehybrid_score

    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    for x, y, filenames in tqdm(loader, desc="Dataset Iteration"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        assert len(x) == len(y) == len(filenames), "Batch size mismatch in DataLoader"

        # Score: (batch size, H, W)
        # Logits: (batch size, 19, H, W)
        score, logits = get_model_output(model, score_func, x)
        _, preds = logits[:, :19, :, :].max(dim=1)

        if args.dump_logits:
            logit_path = out_path / 'logits'
            logit_path.mkdir(exist_ok=True)
            for batch in range(len(filenames)):
                rel_filepath = Path(filenames[batch]).relative_to(dataset.img_root)
                dst_folder = logit_path / dataset_name / model_name / rel_filepath.parent
                dst_folder.mkdir(exist_ok=True, parents=True)
                destfile_logits = dst_folder / f"{rel_filepath.stem}_logits.npz"
                np.savez_compressed(destfile_logits, logits=logits[batch].cpu().to(torch.float16).numpy())
        else:
            for batch in range(len(filenames)):
                pred_saveformat = preds[batch].cpu().numpy().astype(np.uint8)
                conf = score[batch]
                conf = 1.0 - ((conf - conf.min()) / (conf.max() - conf.min()))
                conf = (conf * 65535).cpu().numpy().astype(np.uint16)

                filepath = Path(filenames[batch]).relative_to(args.datasets_folder)
                dst_folder = out_path / model_name / filepath.parent
                dst_folder.mkdir(exist_ok=True, parents=True)
                destfile_pred = dst_folder / f"{filepath.stem}_pred.png"
                destfile_conf = dst_folder / f"{filepath.stem}_conf.png"

                suc = cv2.imwrite(destfile_pred, pred_saveformat)
                if not suc:
                    print(f"Failed to save {destfile_pred}")
                suc = cv2.imwrite(destfile_conf, conf)
                if not suc:
                    print(f"Failed to save {destfile_conf}")


def main():
    # The name of every directory inside args.models_folder is expected to be the model name.
    # Inside a model's folder there should be 2 files (doesn't matter if there are extra stuff).
    # these 2 files are: config.yaml and [model_final.pth or model_final.pkl]

    models_list = os.listdir(args.models_folder)
    models_list = [m for m in models_list if os.path.isdir(
        os.path.join(args.models_folder, m))]
    print(args.models_folder)
    if args.model_mode == 'selective':
        models_list = [m for m in models_list if m in args.selected_models]

    if len(models_list) == 0:
        raise ValueError(
            "Number of models chosen is 0, either model_logs folder is empty or no models were selected")

    print("Generating predictions for the following Models:")
    [print(m) for m in models_list]
    print("-----------------------")

    for model_name in models_list:
        experiment_path = os.path.join(args.models_folder, model_name)

        config_path = os.path.join(experiment_path, 'config.yaml')
        model_path = os.path.join(experiment_path, 'model_final.pth')

        if not os.path.exists(model_path):
            model_path = os.path.join(
                'model_logs', model_name, 'model_final.pkl')

            if not os.path.exists(model_path):
                print("Model path does not exist, skipping")
                continue

        model = get_model(config_path=config_path, model_path=model_path)

        for dataset_name, dataset in dataset_group:
            generate_predictions(model, dataset, dataset_name, model_name, args.out_path)



if __name__ == '__main__':
    main()
