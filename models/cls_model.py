import torch
import torch.nn as nn


class DualStream_LoRA_CLIP_Classifier(nn.Module):
    def __init__(self, lora_clip_model, num_classes, intermediate_feature_dim=512):
        """
        Args:
            lora_clip_model: 注入了LoRA的open_clip模型。
            num_classes (int): 最终输出的类别数。
            intermediate_feature_dim (int): 您希望返回的中间特征的维度。
        """
        super().__init__()
        self.lora_image_encoder = lora_clip_model.visual
        # 编码器输出的特征维度
        encoder_output_dim = self.lora_image_encoder.output_dim
        # 两个模态的特征拼接后的维度
        fused_dim = encoder_output_dim * 2
        # *** 核心修改：构建一个与您旧模型类似的、两层的分类头 ***
        self.head = nn.Sequential(
            nn.Linear(fused_dim, intermediate_feature_dim),
            nn.ReLU(),
            nn.Linear(intermediate_feature_dim, num_classes)
        )
        # 为了方便地获取中间特征，我们也可以将它们分开定义
        self.fc1 = nn.Linear(fused_dim, intermediate_feature_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_feature_dim, num_classes)
    def forward(self, ir_image, vi_image):
        # 1. 分别为IR和VI图像提取特征
        ir_features = self.lora_image_encoder(ir_image)
        vi_features = self.lora_image_encoder(vi_image)
        # 2. 晚期融合：将两个特征向量沿最后一个维度拼接
        fused_features = torch.cat([ir_features, vi_features], dim=-1)
        # 3. 通过分类头的第一部分，得到您需要的中间特征
        intermediate_feature = self.relu(self.fc1(fused_features))
        # 4. 通过分类头的第二部分，得到最终的分类结果 (logits)
        output_logits = self.fc2(intermediate_feature)
        # 5. 返回最终的logits和中间特征
        return output_logits, intermediate_feature