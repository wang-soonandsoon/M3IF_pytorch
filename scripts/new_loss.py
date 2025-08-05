# 在你的 losses.py 文件或主脚本开头添加这个类
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection or Segmentation.

    Args:
        gamma (float): Focusing parameter. Default: 2.0.
        alpha (float or list/tensor): Weighting factor. Default: None.
        reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'.
                         Default: 'mean'. !! 对于你的omega_m用法，必须用 'none' !!
        num_classes (int, Optional): Number of classes. Needed if alpha is list/tensor.
        eps (float): Small value to avoid log(0). Default: 1e-7.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', num_classes=None, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction # <--- 这个参数等下初始化时很重要！
        self.eps = eps
        self.num_classes = num_classes

        if isinstance(alpha, (list, tuple, torch.Tensor)):
            assert num_classes is not None, "num_classes must be provided if alpha is a list/tensor"
            assert len(alpha) == num_classes, "alpha must have length equal to num_classes"
            self.alpha = torch.tensor(alpha)
        elif alpha is not None and not isinstance(alpha, float):
             raise TypeError("alpha must be float, list, tuple, or torch.Tensor")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits (N, C, H, W).
            targets (torch.Tensor): Ground truth labels (N, H, W, dtype=torch.long).
        Returns:
            torch.Tensor: Focal loss.
        """
        N, C, H, W = inputs.shape
        assert targets.shape == (N, H, W), f"Target shape mismatch {targets.shape} vs {(N, H, W)}"
        assert targets.dtype == torch.long, "Targets must be torch.long"

        log_prob = F.log_softmax(inputs, dim=1)
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1) # (N, H, W)
        p_t = log_p_t.exp()
        focal_term = (1 - p_t).pow(self.gamma)
        ce_term = -log_p_t
        loss_per_pixel = focal_term * ce_term # (N, H, W)

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)
            if isinstance(self.alpha, torch.Tensor):
                alpha_weights = alpha_t.gather(0, targets.flatten()).view(N, H, W)
                loss_per_pixel = alpha_weights * loss_per_pixel
            elif isinstance(self.alpha, float):
                 # 对于多分类，简单的alpha缩放可能不够精细，通常用类别权重
                 # 这里简化处理，直接乘以alpha，如果需要更复杂逻辑需要修改
                 loss_per_pixel = self.alpha * loss_per_pixel

        # 应用 reduction
        if self.reduction == 'mean':
            loss = loss_per_pixel.mean()
        elif self.reduction == 'sum':
            loss = loss_per_pixel.sum()
        elif self.reduction == 'none':
            loss = loss_per_pixel # <--- 返回每个像素的损失
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

        return loss

