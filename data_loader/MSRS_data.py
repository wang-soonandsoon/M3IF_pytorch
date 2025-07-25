import os
import random
import shutil

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F # 确保导入 functional
class MSRSTrainDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 label_transform=None,
                 vis_degradation_root_dir=None, # 退化可见光根目录
                 ir_degradation_root_dir=None,  # 退化红外光根目录
                 vis_degradation_types=None,    # 可见光退化类型列表
                 ir_degradation_types=None,     # 红外光退化类型列表
                 clip_input_source='vis_only' ,  # 'vis_only' 或 'degraded_modality'
                 crop_size=448  # 新增：定义裁切大小  # 新增参数：是否随机裁切
                ):
        """
        Args:
            root_dir (str): 包含 'vi', 'ir', 'label' 子文件夹的根目录.
            transform (callable, optional): 应用于图像的转换 (VIS, IR, Degraded).
            label_transform (callable, optional): 应用于标签的转换 (通常不需要).
            vis_degradation_root_dir (str, optional): 退化可见光图像的根目录.
                                                    默认为 root_dir/de_vi.
            ir_degradation_root_dir (str, optional): 退化红外光图像的根目录.
                                                   默认为 root_dir/de_ir.
            vis_degradation_types (list, optional): 可见光退化类型的列表 (文件夹名).
                                                    默认为 ["blur", "haze", "rain", "random_noise", "light"].
            ir_degradation_types (list, optional): 红外光退化类型的列表 (文件夹名).
                                                   默认为 ["stripe_noise", "other_ir_degradations"].
                                                   ***你需要根据实际情况提供这个列表***
            clip_input_source (str): 'vis_only' 或 'degraded_modality'. 控制 clip 输入的来源.
        """
        self.root_dir = root_dir
        self.transform = transform
        # label_transform 通常在加载后直接处理，这里保留以防万一
        self.label_transform = label_transform # 但在 __getitem__ 中我们直接处理了

        # --- 配置退化路径 ---
        self.vis_degradation_root_dir = vis_degradation_root_dir or os.path.join(root_dir, "de_vi")
        self.ir_degradation_root_dir = ir_degradation_root_dir or os.path.join(root_dir, "de_ir")

        # --- 配置退化类型 ---
        self.vis_degradation_types = vis_degradation_types or [
            "blur", "haze", "rain", "random_noise", "light"
        ]
        # !!! 重要: 请根据你的实际文件夹结构更新此列表 !!!
        self.ir_degradation_types = ir_degradation_types or ["stripe_noise"] # 示例，需要你提供

        # --- 配置 CLIP 输入来源 ---
        if clip_input_source not in ['vis_only', 'degraded_modality']:
            raise ValueError("clip_input_source must be 'vis_only' or 'degraded_modality'")
        self.clip_input_source = clip_input_source

        # --- 收集文件 ---
        vi_folder = os.path.join(root_dir, "vi")
        ir_folder = os.path.join(root_dir, "ir")
        label_folder = os.path.join(root_dir, "label")

        if not os.path.isdir(vi_folder) or not os.path.isdir(ir_folder) or not os.path.isdir(label_folder):
             raise FileNotFoundError(f"Required subfolders ('vi', 'ir', 'label') not found in {root_dir}")

        self.vi_files = sorted([f for f in os.listdir(vi_folder) if os.path.isfile(os.path.join(vi_folder, f))])
        self.ir_files = sorted([f for f in os.listdir(ir_folder) if os.path.isfile(os.path.join(ir_folder, f))])
        self.label_files = sorted([f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))])
        self.crop_size = crop_size  # 存储裁切大小
        # 检查文件列表是否匹配且非空
        if not self.vi_files:
            raise FileNotFoundError(f"No image files found in {vi_folder}")
        if not (self.vi_files == self.ir_files == self.label_files):
            # 可以加入更详细的错误报告，比如哪些文件不匹配
            print(f"Warning: File lists in vi, ir, or label directories might not perfectly match.")
            # 取交集可能更安全，但这里我们先按原始逻辑要求严格匹配
            # self.common_files = sorted(list(set(self.vi_files) & set(self.ir_files) & set(self.label_files)))
            # if not self.common_files:
            #    raise ValueError("No common files found across vi, ir, and label directories.")
            # self.vi_files = self.ir_files = self.label_files = self.common_files
            assert self.vi_files == self.ir_files, "Visible (vi) and Infrared (ir) images must have the same filenames."
            assert self.vi_files == self.label_files, "Visible (vi) and Label images must have the same filenames."


        # --- 定义 CLIP 的特定 Transform ---
        # (保持和你原来代码一致)
        self.cliptransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 根据需要取消注释归一化
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        # --- 预检查退化文件夹是否存在 (可选但推荐) ---
        self._check_degradation_folders()

    def _check_degradation_folders(self):
        """检查所有必需的退化文件夹是否存在"""
        print("Checking VIS degradation folders...")
        for deg_type in self.vis_degradation_types:
            folder = os.path.join(self.vis_degradation_root_dir, deg_type)
            if not os.path.isdir(folder):
                print(f"Warning: VIS degradation folder not found: {folder}")
                # raise FileNotFoundError(f"VIS degradation folder not found: {folder}")
        print("Checking IR degradation folders...")
        for deg_type in self.ir_degradation_types:
             folder = os.path.join(self.ir_degradation_root_dir, deg_type)
             if not os.path.isdir(folder):
                 print(f"Warning: IR degradation folder not found: {folder}")
                 # raise FileNotFoundError(f"IR degradation folder not found: {folder}")

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        filename = self.vi_files[idx]

        # --- 1. 加载干净的图像和标签 ---
        clean_vi_path = os.path.join(self.root_dir, "vi", filename)
        clean_ir_path = os.path.join(self.root_dir, "ir", filename)
        label_path = os.path.join(self.root_dir, "label", filename)

        try:
            clean_vi_pil = Image.open(clean_vi_path).convert('RGB')
            clean_ir_pil = Image.open(clean_ir_path).convert('RGB') # 确保IR也是RGB
            label_pil = Image.open(label_path) # 标签通常是单通道灰度图
        except FileNotFoundError as e:
            print(f"Error loading base image/label for index {idx}, filename {filename}: {e}")
            # 可以返回 None 或引发异常，或尝试跳过（不推荐）
            # 这里我们重新引发异常
            raise e
        except Exception as e:
             print(f"Error opening image/label file {filename}: {e}")
             raise e


        # 处理标签: 转为 Long Tensor
        label_np = np.array(label_pil)
        # 可以在这里应用 label_transform (如果定义了)
        # if self.label_transform:
        #    label_np = self.label_transform(label_np) # 假设transform接受numpy
        label_tensor = torch.as_tensor(label_np, dtype=torch.long)
        # 确保标签是 [H, W] 或 [1, H, W] 格式，取决于你的模型需求
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0) # 转为 [1, H, W]

        # --- 2. 决定退化哪个模态 ---
        degrade_modality = random.choice(['vis', 'ir'])

        # 初始化输入图像为干净图像 (PIL格式)
        input_vis_pil = clean_vi_pil.copy()
        input_ir_pil = clean_ir_pil.copy()
        degraded_source_pil = None # 用于记录哪个PIL图像是退化源，供CLIP使用

        # --- 3. 加载选择的退化图像 ---
        try:
            if degrade_modality == 'vis':
                if not self.vis_degradation_types:
                     print(f"Warning: No VIS degradation types specified for {filename}. Using clean VIS.")
                     degraded_source_pil = input_vis_pil # CLIP 使用 (未退化的) VIS
                else:
                    vis_deg_type = random.choice(self.vis_degradation_types)
                    degraded_vis_path = os.path.join(self.vis_degradation_root_dir, vis_deg_type, filename)
                    if not os.path.exists(degraded_vis_path):
                         print(f"Warning: Degraded VIS image not found: {degraded_vis_path}. Using clean VIS.")
                         # degraded_source_pil = input_vis_pil # 保持 clean vis
                         # 保持 input_vis_pil 为 clean_vi_pil.copy()
                         degraded_source_pil = input_vis_pil # CLIP 仍基于 VIS 输入
                    else:
                        input_vis_pil = Image.open(degraded_vis_path).convert('RGB')
                        degraded_source_pil = input_vis_pil # 退化源是加载的这个图像
                # IR 保持干净 (input_ir_pil 已经是 clean_ir_pil.copy())

            elif degrade_modality == 'ir':
                if not self.ir_degradation_types:
                    print(f"Warning: No IR degradation types specified for {filename}. Using clean IR.")
                    # VIS 保持干净 (input_vis_pil 已经是 clean_vi_pil.copy())
                    degraded_source_pil = input_vis_pil # CLIP 如果是 vis_only, 使用 clean vis
                else:
                    ir_deg_type = random.choice(self.ir_degradation_types)
                    degraded_ir_path = os.path.join(self.ir_degradation_root_dir, ir_deg_type, filename)
                    if not os.path.exists(degraded_ir_path):
                        print(f"Warning: Degraded IR image not found: {degraded_ir_path}. Using clean IR.")
                        # degraded_source_pil = input_vis_pil # CLIP 如果是 vis_only, 使用 clean vis
                        # 保持 input_ir_pil 为 clean_ir_pil.copy()
                    else:
                        input_ir_pil = Image.open(degraded_ir_path).convert('RGB')
                        degraded_source_pil = input_ir_pil # 退化源是加载的这个图像
                # VIS 保持干净 (input_vis_pil 已经是 clean_vi_pil.copy())

        except FileNotFoundError as e:
             print(f"Error loading degradation image for {filename} ({degrade_modality}): {e}. Using clean image instead.")
             # 如果加载退化图像失败，确保 degraded_source_pil 指向对应的干净图像
             if degrade_modality == 'vis':
                 degraded_source_pil = clean_vi_pil.copy()
             else: # degrade_modality == 'ir'
                 # 如果 clip_input_source == 'degraded_modality'，理论上应该指向 clean_ir_pil
                 # 但下面 clip_input_base 的逻辑会处理
                 pass # degraded_source_pil 会在下面根据 clip_input_source 决定
        except Exception as e:
             print(f"Error opening degradation image file {filename} ({degrade_modality}): {e}. Using clean image instead.")
             # 同上处理
             if degrade_modality == 'vis':
                 degraded_source_pil = clean_vi_pil.copy()
             else:
                 pass


        # --- 4. 确定 CLIP 输入的来源 (PIL 格式) ---
        clip_input_base_pil = None
        if self.clip_input_source == 'vis_only':
            # 始终使用当前的 VIS 输入（可能是干净的，也可能是退化的）
            clip_input_base_pil = input_vis_pil.copy()
        elif self.clip_input_source == 'degraded_modality':
            # 使用被选择进行退化的那个模态的图像
            # 如果退化加载失败，degraded_source_pil 可能为 None 或指向干净图像
            if degraded_source_pil is None:
                 # 这种情况理论上不应发生，除非上面 try-except 逻辑有问题
                 # 或退化类型列表为空。作为后备，使用 VIS 输入。
                 print(f"Warning: degraded_source_pil is None for {filename}. Defaulting CLIP input to current VIS.")
                 clip_input_base_pil = input_vis_pil.copy()
            else:
                 clip_input_base_pil = degraded_source_pil.copy() # 使用记录的退化源
        # 获取原始图像尺寸 (假设 clean_vi_pil 存在且与其他图像尺寸相同)
        width, height = clean_vi_pil.size

        # 检查图像尺寸是否足够大以进行裁切
        if width < self.crop_size or height < self.crop_size:
            # 如果图像太小，可以选择：
            # 1. 跳过这个样本 (不推荐，除非数据很多且此情况很少)
            # 2. 对图像进行填充 (Padding) 到至少 crop_size
            # 3. 调整裁切大小 (不推荐，会导致 batch 内尺寸不一)
            # 4. 报错
            # 这里我们选择填充到 crop_size (如果需要)
            padding_left = max(0, (self.crop_size - width) // 2)
            padding_right = max(0, self.crop_size - width - padding_left)
            padding_top = max(0, (self.crop_size - height) // 2)
            padding_bottom = max(0, self.crop_size - height - padding_top)

            # 定义填充操作 (对 PIL Image)
            # 'constant' 表示用 value 填充，对于 RGB 通常是 0 (黑色)
            # 对于标签 (灰度)，也是 0
            padding = (padding_left, padding_top, padding_right, padding_bottom)

            input_vis_pil = F.pad(input_vis_pil, padding, fill=0, padding_mode='constant')
            input_ir_pil = F.pad(input_ir_pil, padding, fill=0, padding_mode='constant')
            clean_vi_pil = F.pad(clean_vi_pil, padding, fill=0, padding_mode='constant')
            clean_ir_pil = F.pad(clean_ir_pil, padding, fill=0, padding_mode='constant')
            label_pil = F.pad(label_pil, padding, fill=0, padding_mode='constant') # 标签也填充

            # 更新填充后的尺寸
            width, height = input_vis_pil.size # 现在至少是 crop_size

        # 获取随机裁切参数 (左上角坐标)
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)

        # 应用相同的裁切参数到所有需要同步的 PIL 图像
        input_vis_pil = F.crop(input_vis_pil, top, left, self.crop_size, self.crop_size)
        input_ir_pil = F.crop(input_ir_pil, top, left, self.crop_size, self.crop_size)
        clean_vi_pil = F.crop(clean_vi_pil, top, left, self.crop_size, self.crop_size)
        clean_ir_pil = F.crop(clean_ir_pil, top, left, self.crop_size, self.crop_size)
        label_pil = F.crop(label_pil, top, left, self.crop_size, self.crop_size) # 标签也裁切

        # --- 5. 应用 Transforms ---
        # 对输入和干净图像应用主 transform
        if self.transform:
            # 应用于最终的输入图像
            input_vis = self.transform(input_vis_pil)
            input_ir = self.transform(input_ir_pil)
            # 应用于干净的 Ground Truth 图像
            clean_vis = self.transform(clean_vi_pil)
            clean_ir = self.transform(clean_ir_pil)
        else:
            # 如果没有 transform，至少转成 Tensor
            to_tensor = transforms.ToTensor()
            input_vis = to_tensor(input_vis_pil)
            input_ir = to_tensor(input_ir_pil)
            clean_vis = to_tensor(clean_vi_pil)
            clean_ir = to_tensor(clean_ir_pil)

        # 对 CLIP 输入应用 cliptransform
        # 确保 clip_input_base_pil 不是 None
        if clip_input_base_pil is None:
             print(f"Error: clip_input_base_pil is None before cliptransform for {filename}. Using fallback.")
             # 提供一个后备，例如使用 clean_vi_pil
             clip_input_base_pil = clean_vi_pil.copy() # 或者 input_vis_pil.copy()

        clip_input = self.cliptransform(clip_input_base_pil)
        # --- 6. 处理标签 (现在是对裁切后的标签) ---
        label_np = np.array(label_pil) # 从裁切后的 PIL 转 Numpy
        # 可以在这里应用 label_transform (如果需要)
        # if self.label_transform: label_np = self.label_transform(label_np)
        label_tensor = torch.as_tensor(label_np, dtype=torch.long) # 转 Tensor
        # 确保标签是 [H, W] 或 [1, H, W]，现在 H, W 应该是 crop_size
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0) # 转为 [1, crop_size, crop_size]

        # --- 6. 返回结果 ---
        # 返回: (输入VIS, 输入IR, 干净VIS, 干净IR, 标签, CLIP输入, 文件名)
        return input_vis, input_ir, clean_vis, clean_ir, label_tensor, clip_input, filename


class MatchingImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.vi_files = os.listdir(os.path.join(root_dir, "vi"))
        self.ir_files = os.listdir(os.path.join(root_dir, "ir"))
        # self.label_files = os.listdir(os.path.join(root_dir, "label"))

        assert len(self.vi_files) == len(self.ir_files), \
            "Number of images in 'vi' and 'ir' directories must match."

    def __len__(self):
        return len(self.vi_files)

    def __getitem__(self, idx):
        vi_image_path = os.path.join(self.root_dir, "vi", self.vi_files[idx])
        ir_image_path = os.path.join(self.root_dir, "ir", self.ir_files[idx])
        # label_image_path = os.path.join(self.root_dir, "label", self.label_files[idx])

        vi_image = Image.open(vi_image_path)
        ir_image = Image.open(ir_image_path).convert('RGB')

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)
            # label_image = self.transform(label_image)

        return vi_image, ir_image

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 创建临时目录和虚拟图片
    temp_root_dir = "./temp_dataset"
    os.makedirs(os.path.join(temp_root_dir, "vi"), exist_ok=True)
    os.makedirs(os.path.join(temp_root_dir, "ir"), exist_ok=True)
    os.makedirs(os.path.join(temp_root_dir, "label"), exist_ok=True)

    # 创建退化目录
    temp_de_vis_dir = os.path.join(temp_root_dir, "de_vi")
    temp_de_ir_dir = os.path.join(temp_root_dir, "de_ir")
    os.makedirs(os.path.join(temp_de_vis_dir, "blur"), exist_ok=True)
    os.makedirs(os.path.join(temp_de_ir_dir, "stripe_noise"), exist_ok=True)

    num_images = 5 # 虚拟图片数量
    img_size = (500, 500) # 原始图片尺寸，大于 crop_size
    crop_size = 448 # 裁切尺寸

    print(f"创建 {num_images} 张虚拟图片...")
    for i in range(num_images):
        filename = f"{i:04d}.png"
        # 创建一个全黑的 PIL 图像
        dummy_image = Image.new('RGB', img_size, color = 'black')
        dummy_label = Image.new('L', img_size, color = 0) # 标签为灰度图

        dummy_image.save(os.path.join(temp_root_dir, "vi", filename))
        dummy_image.save(os.path.join(temp_root_dir, "ir", filename))
        dummy_label.save(os.path.join(temp_root_dir, "label", filename))

        # 创建退化图片
        dummy_image.save(os.path.join(temp_de_vis_dir, "blur", filename))
        dummy_image.save(os.path.join(temp_de_ir_dir, "stripe_noise", filename))
    print("虚拟图片创建完成。")

    # 2. 定义一个简单的 transform
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 根据需要添加归一化
    ])

    # 3. 实例化数据集
    print("\n实例化 MSRSTrainDataset...")
    try:
        dataset = MSRSTrainDataset(
            root_dir=temp_root_dir,
            transform=data_transform,
            vis_degradation_types=["blur"], # 仅使用我们创建的退化类型
            ir_degradation_types=["stripe_noise"],
            clip_input_source='degraded_modality', # 测试 'degraded_modality' 模式
            crop_size=crop_size
        )
        print(f"数据集实例化成功，包含 {len(dataset)} 个样本。")

        # 4. 测试 __getitem__
        print("\n测试从数据集中获取样本...")
        for i in range(min(3, len(dataset))): # 获取前3个样本进行测试
            print(f"\n--- 获取样本 {i} ---")
            input_vis, input_ir, clean_vis, clean_ir, label_tensor, clip_input, filename = dataset[i]

            print(f"文件名: {filename}")
            print(f"输入可见光图像形状 (input_vis): {input_vis.shape}")
            print(f"输入红外光图像形状 (input_ir): {input_ir.shape}")
            print(f"干净可见光图像形状 (clean_vis): {clean_vis.shape}")
            print(f"干净红外光图像形状 (clean_ir): {clean_ir.shape}")
            print(f"标签图像形状 (label_tensor): {label_tensor.shape}")
            print(f"CLIP 输入图像形状 (clip_input): {clip_input.shape}")

            # 验证裁切尺寸
            assert input_vis.shape[1] == crop_size and input_vis.shape[2] == crop_size, "input_vis 裁切尺寸不正确"
            assert input_ir.shape[1] == crop_size and input_ir.shape[2] == crop_size, "input_ir 裁切尺寸不正确"
            assert label_tensor.shape[1] == crop_size and label_tensor.shape[2] == crop_size, "label_tensor 裁切尺寸不正确"
            assert clip_input.shape[1] == 224 and clip_input.shape[2] == 224, "clip_input 尺寸不正确"

        print("\n所有测试样本获取成功，形状符合预期。")

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    finally:
        # 5. 清理临时目录
        print(f"\n清理临时目录: {temp_root_dir}...")
        if os.path.exists(temp_root_dir):
            shutil.rmtree(temp_root_dir)
            print("临时目录清理完成。")

