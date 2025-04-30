"""
CompletionFormer
======================================================================

Custom FLSea Dataset for text-based splits
Reads sample list from a text file with three paths per line:
1) RGB image (TIFF)
2) Ground truth depth map (TIFF)
3) Sparse depth CSV/TXT file with rows: row,column,depth
"""

import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from . import BaseDataset

warnings.filterwarnings("ignore", category=UserWarning)

def generate_feature_map_for_ga(
    feature_fp,
    original_height=480,
    original_width=640,
    new_height=336,
    new_width=448,
    inverse_depth=False
):
    # Determine delimiter based on file extension
    ext = os.path.splitext(feature_fp)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(feature_fp, delimiter=' ', header=0, names=['row','column','depth'])
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT files are supported.")

    # Initialize empty sparse depth map
    sparse_depth_map = np.zeros((new_height, new_width), dtype=np.float32)

    # Compute scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Populate sparse map
    for _, r in df.iterrows():
        pr = int(r['row'] * scale_y)
        pc = int(r['column'] * scale_x)
        d = float(r['depth'])
        if inverse_depth:
            d = 1.0 / d
        if 0 <= pr < new_height and 0 <= pc < new_width:
            sparse_depth_map[pr, pc] = d

    # Add channel dimension and return
    return sparse_depth_map[..., np.newaxis]

class FLSea(BaseDataset):
    def __init__(self, args, mode):
        super(FLSea, self).__init__(args, mode)
        self.args = args
        self.mode = mode
        if mode not in ['train','val','test']:
            raise NotImplementedError(f"Mode '{mode}' not supported.")

        # Define target size and crop
        self.height, self.width = 336, 448
        self.crop_size = (228, 304)

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            1296.666758476217,
            1300.831316354508,
            501.50386149846,
            276.161712082695
        ])

        self.augment = args.augment

        # Load sample list from text file
        with open(self.args.split_txt, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.sample_list = [ln.split() for ln in lines]

        # Define transforms
        self.t_rgb = T.Compose([
            T.Resize((336,448)),
            # T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ])
        self.t_dep = T.Compose([
            T.Resize((336,448)),
            # T.CenterCrop(self.crop_size),
            self.ToNumpy(),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb_fp, gt_fp, sparse_fp = self.sample_list[idx]

        # Load RGB image
        rgb = Image.open(rgb_fp).convert('RGB')
        rgb = self.t_rgb(rgb)

        # Load GT depth map
        depth_img = Image.open(gt_fp)
        if depth_img.mode != 'F':
            depth_img = depth_img.convert('F')
        dep = self.t_dep(depth_img)

        # Camera intrinsics
        K = self.K.clone()

        # Generate sparse depth map
        sparse_map = generate_feature_map_for_ga(
            sparse_fp,
            original_height=240,
            original_width=320,
            new_height=336,
            new_width=448,
            inverse_depth=False
        )
        dep_sp = torch.from_numpy(
            np.transpose(sparse_map, (2, 0, 1))
        ).type_as(dep)

        return {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}
