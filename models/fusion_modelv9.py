import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math  # For the dummy ECA block

from .MItransformer import HeterogeneousTransformerBlock
class IndependentSpatialGatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(IndependentSpatialGatedFusionBlock, self).__init__()
        reduction = max(4, channels // 8)
        self.channels = channels
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, v, i):
        combined = torch.cat([v, i], dim=1)
        gates = self.attention(combined)
        gate_v, gate_i = torch.split(gates, self.channels, dim=1)
        gated_fusion = gate_v * v + gate_i * i
        gated_fusion = self.final_conv(gated_fusion)
        return gated_fusion


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class encoder(nn.Module):
    def __init__(self, d_text=512):
        super(encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, stride=1, padding=1, kernel_size=3)
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=16, d_text=d_text, num_experts=8, top_k=2)
        self.down1 = Downsample(n_feat=16)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.down2 = Downsample(n_feat=32)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.down3 = Downsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.down4 = Downsample(n_feat=128)

    def forward(self, x, route_feature, task_id):
        device = x.device
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}
        x1_ = self.conv(x)
        x1_moe, aux_losses1 = self.moe_vit_block1(x1_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        x2_ = self.down1(x1_moe)
        x2_moe, aux_losses2 = self.moe_vit_block2(x2_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        x3_ = self.down2(x2_moe)
        x3_moe, aux_losses3 = self.moe_vit_block3(x3_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        x4_ = self.down3(x3_moe)
        x4_moe, aux_losses4 = self.moe_vit_block4(x4_, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        x5_ = self.down4(x4_moe)
        return x1_, x2_, x3_, x4_, x5_, total_aux_losses


class decoder(nn.Module):
    def __init__(self, d_text=512):
        super(decoder, self).__init__()
        self.moe_vit_block1 = HeterogeneousTransformerBlock(dim=256, d_text=d_text, num_experts=8, top_k=2)
        self.up1 = Upsample(n_feat=256)
        self.moe_vit_block2 = HeterogeneousTransformerBlock(dim=128, d_text=d_text, num_experts=8, top_k=2)
        self.up2 = Upsample(n_feat=128)
        self.moe_vit_block3 = HeterogeneousTransformerBlock(dim=64, d_text=d_text, num_experts=8, top_k=2)
        self.up3 = Upsample(n_feat=64)
        self.moe_vit_block4 = HeterogeneousTransformerBlock(dim=32, d_text=d_text, num_experts=8, top_k=2)
        self.up4 = Upsample(n_feat=32)

    def forward(self, x1, x2, x3, x4, x5, route_feature, task_id):
        device = x5.device
        total_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                            'mi_loss': torch.tensor(0.0, device=device)}
        y5_moe, aux_losses1 = self.moe_vit_block1(x5, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses1.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses1.get('mi_loss', 0.0)
        y4_up = self.up1(y5_moe)
        y4 = y4_up + x4
        y4_moe, aux_losses2 = self.moe_vit_block2(y4, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses2.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses2.get('mi_loss', 0.0)
        y3_up = self.up2(y4_moe)
        y3 = y3_up + x3
        y3_moe, aux_losses3 = self.moe_vit_block3(y3, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses3.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses3.get('mi_loss', 0.0)
        y2_up = self.up3(y3_moe)
        y2 = y2_up + x2
        y2_moe, aux_losses4 = self.moe_vit_block4(y2, route_feature, task_id=task_id)
        total_aux_losses['standard_moe_loss'] += aux_losses4.get('standard_moe_loss', 0.0)
        total_aux_losses['mi_loss'] += aux_losses4.get('mi_loss', 0.0)
        y1_up = self.up4(y2_moe)
        y1 = y1_up + x1
        return y1, total_aux_losses


# =============================================================================
# --- NEW: Reusable Conv-GELU-Conv Head Module ---
# =============================================================================
class ConvGeluConvHead(nn.Module):
    """
    一个可重用的头模块，实现了 '卷积-GELU-卷积' 结构。
    - 第一个卷积 (3x3) 用于捕捉空间上下文特征。
    - GELU 提供了平滑的非线性激活。
    - 第二个卷积 (1x1) 用于将特征投影到最终的输出通道。
    """

    def __init__(self, in_channels, intermediate_channels, out_channels, kernel_size=3):
        super(ConvGeluConvHead, self).__init__()
        self.head = nn.Sequential(
            # 第一个卷积层，用于特征提取和通道变换
            nn.Conv2d(in_channels, intermediate_channels,
                      kernel_size=kernel_size, padding=(kernel_size // 2), bias=False),
            # GELU 激活函数
            nn.GELU(),
            # 第二个卷积层 (1x1)，用于生成最终输出
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


# =============================================================================
# --- MODIFIED: Main MSGFusion Model ---
# =============================================================================
class M3IF(nn.Module):
    def __init__(self, n_class=9, d_text=512):
        super(M3IF, self).__init__()
        self.n_class = n_class
        self.encoder = encoder(d_text=d_text)
        self.decode = decoder(d_text=d_text)

        # Gated Fusion Blocks (保持不变)
        self.fusion_block1 = IndependentSpatialGatedFusionBlock(channels=16)
        self.fusion_block2 = IndependentSpatialGatedFusionBlock(channels=32)
        self.fusion_block3 = IndependentSpatialGatedFusionBlock(channels=64)
        self.fusion_block4 = IndependentSpatialGatedFusionBlock(channels=128)
        self.fusion_block5 = IndependentSpatialGatedFusionBlock(channels=256)

        # --- MODIFICATION START: Standardized Heads ---
        # 使用新的 ConvGeluConvHead 模块来定义所有的头
        # 这样做可以统一架构，同时通过调整 intermediate_channels 来控制每个头的复杂度

        # 融合头
        # self.fusion_head = nn.Sequential(
        #     # 1. 第一个卷积层，将通道从16扩展到32
        #     nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False),
        #     nn.GELU(),
        #     # 2. 新增的中间层，作为缓冲，将通道从32压缩到16
        #     nn.Conv2d(24, 16, kernel_size=3, padding=1, bias=False),
        #     nn.GELU(),
        #     # 3. 最后的1x1卷积，将通道从16投影到最终的3（RGB）
        #     nn.Conv2d(16, 3, kernel_size=1)
        # )
        self.fusion_head = nn.Sequential(
                    # 1. 第一个卷积层，将通道从16扩展到32
                    ConvGeluConvHead(
                        in_channels=16,
                        intermediate_channels=24,  # 为分割任务设置一个中等大小的中间层
                        out_channels=8
                    ),
                    nn.GELU(),
                    # 3. 最后的1x1卷积，将通道从16投影到最终的3（RGB）
                    nn.Conv2d(8, 3, kernel_size=1)
                )
        # 分割头
        self.seg_head = ConvGeluConvHead(
            in_channels=16,
            intermediate_channels=16,  # 为分割任务设置一个中等大小的中间层
            out_channels=self.n_class
        )

        # 复原头 (为复原任务设置一个更轻量级的中间层)
        self.recon_head_vis = ConvGeluConvHead(
            in_channels=16,
            intermediate_channels=16,
            out_channels=3
        )
        self.recon_head_inf = ConvGeluConvHead(
            in_channels=16,
            intermediate_channels=16,
            out_channels=3
        )
        # --- MODIFICATION END ---

    def _reconstruct_and_segment(self, fused_image, route_feature):
        device = fused_image.device
        recon_seg_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                                'mi_loss': torch.tensor(0.0, device=device)}
        rf1, rf2, rf3, rf4, rf5, aux_losses_enc = self.encoder(fused_image, route_feature, task_id=1)
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_enc.get('standard_moe_loss',
                                                                        torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_enc.get('mi_loss', torch.tensor(0.0, device=device))
        y1_decoded_features, aux_losses_dec = self.decode(rf1, rf2, rf3, rf4, rf5, route_feature, task_id=1)
        recon_seg_aux_losses['standard_moe_loss'] += aux_losses_dec.get('standard_moe_loss',
                                                                        torch.tensor(0.0, device=device))
        recon_seg_aux_losses['mi_loss'] += aux_losses_dec.get('mi_loss', torch.tensor(0.0, device=device))

        # 应用新的头模块
        bw_vi = self.recon_head_vis(y1_decoded_features)
        bw_ir = self.recon_head_inf(y1_decoded_features)
        seg_res = self.seg_head(y1_decoded_features)

        return bw_vi, bw_ir, seg_res, recon_seg_aux_losses

    def forward(self, vi, ir, route_feature):
        device = vi.device
        fusion_aux_losses = {'standard_moe_loss': torch.tensor(0.0, device=device),
                             'mi_loss': torch.tensor(0.0, device=device)}
        v1, v2, v3, v4, v5, aux_losses_enc_v = self.encoder(vi, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_v.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_v.get('mi_loss', 0.0)
        i1, i2, i3, i4, i5, aux_losses_enc_i = self.encoder(ir, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_enc_i.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_enc_i.get('mi_loss', 0.0)
        f1 = self.fusion_block1(v1, i1)
        f2 = self.fusion_block2(v2, i2)
        f3 = self.fusion_block3(v3, i3)
        f4 = self.fusion_block4(v4, i4)
        f5 = self.fusion_block5(v5, i5)
        y1_fused, aux_losses_dec_fus = self.decode(f1, f2, f3, f4, f5, route_feature, task_id=0)
        fusion_aux_losses['standard_moe_loss'] += aux_losses_dec_fus.get('standard_moe_loss', 0.0)
        fusion_aux_losses['mi_loss'] += aux_losses_dec_fus.get('mi_loss', 0.0)

        # 应用新的融合头
        fusion_res = self.fusion_head(y1_fused)

        if self.training:
            bw_vi, bw_ir, seg_res, recon_seg_aux_losses = self._reconstruct_and_segment(
                fusion_res, route_feature
            )
            total_aux_losses = {}
            total_aux_losses['standard_moe_loss'] = fusion_aux_losses.get('standard_moe_loss', 0.0) + \
                                                    recon_seg_aux_losses.get('standard_moe_loss', 0.0)
            total_aux_losses['mi_loss'] = fusion_aux_losses.get('mi_loss', 0.0) + \
                                          recon_seg_aux_losses.get('mi_loss', 0.0)
            return fusion_res, seg_res, bw_vi, bw_ir, total_aux_losses
        else:
            return fusion_res, None, None, None, fusion_aux_losses


# =============================================================================
# Test Script
# =============================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    image_vis = torch.rand(1, 3, 256, 256, device=device)
    image_ir = torch.rand(1, 3, 256, 256, device=device)
    route_feature = torch.rand(1, 512, device=device)

    model = M3IF(n_class=9, d_text=512).to(device)

    print("\n--- 打印模型架构 (只显示头部分) ---")
    print("Fusion Head:", model.fusion_head)
    print("Segmentation Head:", model.seg_head)
    print("Reconstruction Head (Vis):", model.recon_head_vis)

    print("\n--- 测试训练模式 (model.train()) ---")
    model.train()
    outputs_train = model(image_vis, image_ir, route_feature)
    fusion_res_train, seg_res_train, bw_vi_train, bw_ir_train, total_aux_losses_train = outputs_train
    print("输出形状:")
    print(f"  融合结果 (fusion_res):      {fusion_res_train.shape}")
    print(f"  分割结果 (seg_res):   {seg_res_train.shape}")
    print(f"  复原可见光 (bw_vi):         {bw_vi_train.shape}")
    print(f"  复原红外光 (bw_ir):        {bw_ir_train.shape}")
    print("\n辅助损失字典 (total_aux_losses):")
    print(f"  字典: {total_aux_losses_train}")

    print("\n--- 测试评估模式 (model.eval()) ---")
    model.eval()
    with torch.no_grad():
        outputs_eval = model(image_vis, image_ir, route_feature)
    fusion_res_eval, seg_res_eval, bw_vi_eval, bw_ir_eval, fusion_aux_losses_eval = outputs_eval
    print("输出值/形状:")
    print(f"  融合结果 (fusion_res):      {fusion_res_eval.shape if fusion_res_eval is not None else 'None'}")
    print(f"  分割结果 (seg_res):   {seg_res_eval}")
    print(f"  复原可见光 (bw_vi):         {bw_vi_eval}")
    print(f"  复原红外光 (bw_ir):        {bw_ir_eval}")
    print("\n辅助损失字典 (fusion_aux_losses):")
    print(f"  字典: {fusion_aux_losses_eval}")

    print("\n测试完成。")
