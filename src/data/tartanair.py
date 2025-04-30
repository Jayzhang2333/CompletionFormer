import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import BaseDataset
import torchvision.transforms as T

# Suppress non-writeable NumPy array and upsampling warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The given NumPy array is not writeable")
warnings.filterwarnings("ignore", message="Default upsampling behavior")

def generate_feature_map_for_ga(
    feature_fp,
    original_height,
    original_width,
    new_height,
    new_width,
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


class TartanAir(BaseDataset):
    def __init__(self, args, mode):
        super(TartanAir, self).__init__(args, mode)
        if mode not in ['train', 'val', 'test']:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        self.args = args
        self.mode = mode
        self.height = 336
        self.width = 448
        self.sparse_point_orig_h = 480
        self.sparse_point_orig_w = 640

        # Camera intrinsics [fx, fy, cx, cy]
        # these are retained for compatibility
        self.K = torch.Tensor([
            320,
            320,
            320,
            240
        ])

        # Load samples from txt file: each line -> rgb_fp, gt_fp, sparse_fp
        self.sample_list = []
        if mode == 'train':
            txt_file = self.args.split_txt_training
        elif mode == 'val':
            txt_file = self.args.split_txt_validation
        else:
            txt_file = self.args.split_txt
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(f"Expected 3 paths per line, got {len(parts)}: {parts}")
                self.sample_list.append({'rgb': parts[0], 'gt': parts[1], 'sp': parts[2]})

        # Define transforms (no augmentation, direct resize)
        self.t_rgb = T.Compose([
            T.Resize((self.height, self.width)),
            T.Lambda(lambda img: img.copy()),  # ensure writeable
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.t_dep = T.Compose([
            T.Resize((self.height, self.width)),
            T.Lambda(lambda img: img.copy()),  # ensure writeable
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]

        # Load RGB image
        rgb = Image.open(sample['rgb']).convert('RGB')

        # Load ground truth depth map from .npy
        gt_np = np.load(sample['gt'])  # expected shape (H, W)
        dep_pil = Image.fromarray(gt_np.astype(np.float32), mode='F')

        # Apply transforms
        rgb_tensor = self.t_rgb(rgb)
        gt_tensor = self.t_dep(dep_pil)

        # Generate sparse depth map
        sparse_np = generate_feature_map_for_ga(
            sample['sp'],
            original_height=self.sparse_point_orig_h,
            original_width=self.sparse_point_orig_w,
            new_height=self.height,
            new_width=self.width,
            inverse_depth=False
        )
        sparse_tensor = torch.from_numpy(sparse_np).permute(2, 0, 1).type_as(gt_tensor)

        # Clone intrinsics for this sample
        K = self.K.clone()

        return {'rgb': rgb_tensor, 'dep': sparse_tensor, 'gt': gt_tensor, 'K': K}
