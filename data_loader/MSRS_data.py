import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class ComprehensivePairedDataset(Dataset):
    """
    一个功能完备的数据集类，用于加载配对的退化/干净/标签数据，
    并为CLIP模型独立准备输入。
    """

    def __init__(self, root_dir, crop_size, split='train', image_transform=None):
        """
        初始化数据集。

        Args:
            root_dir (str): 数据集的根目录。
            crop_size (int): 随机裁切的目标尺寸。
            split (str, optional): 'train', 'val', 'test'等。 默认为 'train'。
            image_transform (callable, optional): 应用于主任务图像(退化/干净)的变换。
        """
        super().__init__()

        self.root_dir = os.path.join(root_dir, split)
        self.crop_size = crop_size
        self.image_transform = image_transform

        # --- 1. 路径定义 ---
        self.vi_dir = os.path.join(self.root_dir, 'vi')
        self.ir_dir = os.path.join(self.root_dir, 'ir')
        self.de_vi_dir = os.path.join(self.root_dir, 'de_vi')
        self.de_ir_dir = os.path.join(self.root_dir, 'de_ir')
        self.label_dir = os.path.join(self.root_dir, 'label')

        # --- 2. 建立文件列表 ---
        if not os.path.isdir(self.de_vi_dir):
            raise FileNotFoundError(f"退化可见光目录不存在: {self.de_vi_dir}")
        self.file_list = sorted(
            [f for f in os.listdir(self.de_vi_dir) if os.path.isfile(os.path.join(self.de_vi_dir, f))])

        # --- 3. 定义CLIP变换 ---
        self.cliptransform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # 官方CLIP通常在模型内部处理归一化，或在训练时应用。
            # 如有需要，可以取消下面的注释。
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # --- 1. 文件名解析 ---
        degraded_filename = self.file_list[index]
        match = re.match(r'(.+)_v\d+(\..+)', degraded_filename)
        base_filename = f"{match.group(1)}{match.group(2)}" if match else degraded_filename

        # --- 2. 加载所有PIL图像 ---
        try:
            # 加载退化图像
            de_vi_pil = Image.open(os.path.join(self.de_vi_dir, degraded_filename)).convert('RGB')
            de_ir_pil = Image.open(os.path.join(self.de_ir_dir, degraded_filename)).convert('RGB')
            # 加载干净图像
            clean_vi_pil = Image.open(os.path.join(self.vi_dir, base_filename)).convert('RGB')
            clean_ir_pil = Image.open(os.path.join(self.ir_dir, base_filename)).convert('RGB')  # 统一转为RGB
            # 加载标签
            label_pil = Image.open(os.path.join(self.label_dir, base_filename)).convert('L')
        except FileNotFoundError as e:
            print(f"加载文件时出错: {e}。")
            return None  # DataLoader会跳过返回None的样本

        # --- 3. 为CLIP准备输入 (使用干净图像，在裁切前) ---
        clip_input_vi = self.cliptransform(clean_vi_pil)
        clip_input_ir = self.cliptransform(clean_ir_pil)
        clip_inputs = [clip_input_vi, clip_input_ir]

        # --- 4. 同步随机裁切 ---
        # 4a. 填充
        width, height = de_vi_pil.size
        if width < self.crop_size or height < self.crop_size:
            padding_left = max(0, (self.crop_size - width) // 2)
            padding_right = max(0, self.crop_size - width - padding_left)
            padding_top = max(0, (self.crop_size - height) // 2)
            padding_bottom = max(0, self.crop_size - height - padding_top)
            padding = (padding_left, padding_top, padding_right, padding_bottom)

            de_vi_pil = F.pad(de_vi_pil, padding, fill=0)
            de_ir_pil = F.pad(de_ir_pil, padding, fill=0)
            clean_vi_pil = F.pad(clean_vi_pil, padding, fill=0)
            clean_ir_pil = F.pad(clean_ir_pil, padding, fill=0)
            label_pil = F.pad(label_pil, padding, fill=0)
            width, height = de_vi_pil.size

        # 4b. 裁切
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)

        de_vi_pil = F.crop(de_vi_pil, top, left, self.crop_size, self.crop_size)
        de_ir_pil = F.crop(de_ir_pil, top, left, self.crop_size, self.crop_size)
        clean_vi_pil = F.crop(clean_vi_pil, top, left, self.crop_size, self.crop_size)
        clean_ir_pil = F.crop(clean_ir_pil, top, left, self.crop_size, self.crop_size)
        label_pil = F.crop(label_pil, top, left, self.crop_size, self.crop_size)

        # --- 5. 对主任务图像应用变换 ---
        if self.image_transform:
            de_vi_tensor = self.image_transform(de_vi_pil)
            de_ir_tensor = self.image_transform(de_ir_pil)
            clean_vi_tensor = self.image_transform(clean_vi_pil)
            clean_ir_tensor = self.image_transform(clean_ir_pil)
        else:
            to_tensor = transforms.ToTensor()
            de_vi_tensor = to_tensor(de_vi_pil)
            de_ir_tensor = to_tensor(de_ir_pil)
            clean_vi_tensor = to_tensor(clean_vi_pil)
            clean_ir_tensor = to_tensor(clean_ir_pil)

        # --- 6. 处理标签，确保为 Long Tensor 且形状正确 ---
        label_np = np.array(label_pil, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_np).long()  # 转为LongTensor
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0)  # 保证形状为 [1, H, W]

        # --- 7. 按指定顺序返回所有数据 ---
        return de_vi_tensor, de_ir_tensor, clean_vi_tensor, clean_ir_tensor, \
            label_tensor, clip_inputs, degraded_filename


if __name__ == '__main__':
    DATASET_ROOT = '/media/gpu/5eb2b5f9-320a-433d-971d-0c5ba1e2c352/FusionDataset/datasetexp/'  # <<<--- 请修改为你的路径
    CROP_SIZE = 448

    # 为主任务图像定义变换
    main_image_transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 [0, 1] 的 Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
    ])

    try:
        # 实例化最终版 Dataset
        final_dataset = ComprehensivePairedDataset(
            root_dir=DATASET_ROOT,
            crop_size=CROP_SIZE,
            split='train',
            image_transform=main_image_transform
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=final_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )

        print(f"数据集大小: {len(final_dataset)}")

        # 获取一个批次的数据并验证
        de_vi_b, de_ir_b, clean_vi_b, clean_ir_b, label_b, clip_b, fname_b = next(iter(data_loader))

        print("\n--- 单个批次数据形状和类型验证 ---")
        print(f"退化可见光 (de_vi) shape:     {de_vi_b.shape}, dtype: {de_vi_b.dtype}")
        print(f"退化红外 (de_ir) shape:       {de_ir_b.shape}, dtype: {de_ir_b.dtype}")
        print(f"干净可见光 (clean_vi) shape:   {clean_vi_b.shape}, dtype: {clean_vi_b.dtype}")
        print(f"干净红外 (clean_ir) shape:     {clean_ir_b.shape}, dtype: {clean_ir_b.dtype}")
        print(
            f"标签 (label) shape:           {label_b.shape}, dtype: {label_b.dtype}")  # 应该为 [B, 1, 256, 256], torch.int64
        print(f"CLIP可见光输入 shape:        {clip_b[0].shape}, dtype: {clip_b[0].dtype}")  # 应该为 [B, 3, 224, 224]
        print(f"CLIP红外输入 shape:          {clip_b[1].shape}, dtype: {clip_b[1].dtype}")  # 应该为 [B, 3, 224, 224]
        print(f"文件名 (batch内第一个):       {fname_b[0]}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")