import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TierHFLLoss(nn.Module):
    """TierHFL损失函数，包含本地和全局损失平衡"""
    def __init__(self, alpha=0.5, lambda_feature=0.1, temperature=0.07, max_cache_size=10):
        super(TierHFLLoss, self).__init__()
        self.alpha = alpha  # 本地损失权重
        self.lambda_feature = lambda_feature  # 特征对齐权重
        self.temperature = temperature  # 对比学习温度参数
        self.ce = nn.CrossEntropyLoss()
        # 添加投影矩阵缓存字典
        self.projection_matrix_cache = {}
        self.max_cache_size = max_cache_size  # 添加缓存大小限制
        self.cache_hits = 0
        self.cache_misses = 0
    
    def forward(self, local_logits, global_logits, local_features, 
                global_features, targets, round_idx=0):
        """计算总损失
        
        Args:
            local_logits: 本地分类器输出
            global_logits: 全局分类器输出
            local_features: 本地特征
            global_features: 全局特征
            targets: 目标标签
            round_idx: 当前训练轮次
            
        Returns:
            总损失，本地损失，全局损失，特征对齐损失
        """
        # 分类损失
        local_loss = self.ce(local_logits, targets)
        global_loss = self.ce(global_logits, targets)
        
        # 特征对齐损失 - 使用简化的方法
        feature_loss = self._feature_alignment(local_features, global_features)
        
        # 根据轮次动态调整权重
        dynamic_alpha = self._adjust_alpha(self.alpha, round_idx)
        dynamic_lambda = self._adjust_lambda(self.lambda_feature, round_idx)
        
        # 总损失计算
        total_loss = dynamic_alpha * local_loss + (1 - dynamic_alpha) * global_loss + \
                    dynamic_lambda * feature_loss
                    
        return total_loss, local_loss, global_loss, feature_loss
    
    # def _feature_alignment(self, local_feat, global_feat):
    #     """改进的特征对齐损失计算，使用缓存的投影矩阵"""
    #     # 确保特征是二维张量
    #     if local_feat.dim() > 2:
    #         local_feat = F.adaptive_avg_pool2d(local_feat, (1, 1))
    #         local_feat = local_feat.view(local_feat.size(0), -1)
        
    #     if global_feat.dim() > 2:
    #         global_feat = F.adaptive_avg_pool2d(global_feat, (1, 1))
    #         global_feat = global_feat.view(global_feat.size(0), -1)
        
    #     # 如果维度不匹配，使用或创建对应的投影矩阵
    #     if local_feat.size(1) != global_feat.size(1):
    #         # 创建维度组合的键
    #         dim_key = (local_feat.size(1), global_feat.size(1))
    #         device_str = str(local_feat.device)
    #         cache_key = f"{dim_key}_{device_str}"
            
    #         # 检查缓存中是否已有该维度组合的投影矩阵
    #         if cache_key in self.projection_matrix_cache:
    #             projection = self.projection_matrix_cache[cache_key]
    #             self.cache_hits += 1
    #         else:
    #             # 选择较小的维度作为共同维度
    #             common_dim = min(local_feat.size(1), global_feat.size(1))
                
    #             # 创建新的投影矩阵并添加到缓存
    #             projection = torch.randn(
    #                 max(local_feat.size(1), global_feat.size(1)), 
    #                 common_dim, 
    #                 device=local_feat.device
    #             )
    #             self.projection_matrix_cache[cache_key] = projection
    #             self.cache_misses += 1
                
    #             # 限制缓存大小
    #             self._clean_cache()
            
    #         # 应用投影
    #         if local_feat.size(1) > global_feat.size(1):
    #             local_feat = torch.matmul(local_feat, projection[:local_feat.size(1), :])
    #         else:
    #             global_feat = torch.matmul(global_feat, projection[:global_feat.size(1), :])
        
    #     # 特征归一化
    #     local_norm = F.normalize(local_feat, dim=1)
    #     global_norm = F.normalize(global_feat, dim=1)
        
    #     # 计算余弦相似度
    #     cos_sim = torch.sum(local_norm * global_norm, dim=1)
        
    #     # 转换为距离损失
    #     return torch.mean(1.0 - cos_sim)

    # 在TierHFLLoss类中添加
    # 修改utils/tierhfl_loss.py中的dynamic_feature_alignment函数
    def dynamic_feature_alignment(self, local_feat, global_feat, round_idx, total_rounds):
        """简化的动态特征对齐方法 - 处理不同维度的特征"""
        # 确保特征是二维
        if local_feat.dim() > 2:
            local_feat = F.adaptive_avg_pool2d(local_feat, (1, 1)).flatten(1)
        if global_feat.dim() > 2:
            global_feat = F.adaptive_avg_pool2d(global_feat, (1, 1)).flatten(1)
        
        # 处理不同维度的特征 - 新增
        local_dim = local_feat.size(1)
        global_dim = global_feat.size(1)
        
        if local_dim != global_dim:
            # 创建临时映射层，将维度较小的特征映射到与较大维度相同
            device = local_feat.device
            if local_dim < global_dim:
                # 将本地特征映射到全局维度
                projection = torch.nn.Linear(local_dim, global_dim, bias=False).to(device)
                # 初始化为接近单位变换
                nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                local_feat = projection(local_feat)
            else:
                # 将全局特征映射到本地维度
                projection = torch.nn.Linear(global_dim, local_dim, bias=False).to(device)
                nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                global_feat = projection(global_feat)
        
        # 标准化特征
        local_norm = F.normalize(local_feat, dim=1)
        global_norm = F.normalize(global_feat, dim=1)
        
        # 计算余弦相似度损失
        alignment_loss = 1.0 - torch.mean(torch.sum(local_norm * global_norm, dim=1))
        
        # 动态强度因子 - 训练早期强，中期最强，后期减弱
        progress = round_idx / max(1, total_rounds)
        if progress < 0.3:
            # 前30%轮次，中等强度
            dynamic_factor = 0.2
        elif progress < 0.7:
            # 中期40%轮次，最强强度
            dynamic_factor = 0.3
        else:
            # 后30%轮次，减弱强度
            dynamic_factor = max(0.05, 0.2 - (progress - 0.7) * 0.5)
        
        return alignment_loss * dynamic_factor
    
    def _clean_cache(self):
        """清理旧的投影矩阵缓存"""
        if len(self.projection_matrix_cache) > self.max_cache_size:
            # 移除最早添加的项
            keys = list(self.projection_matrix_cache.keys())
            for key in keys[:len(keys) - self.max_cache_size]:
                del self.projection_matrix_cache[key]
    
    def _adjust_alpha(self, base_alpha, round_idx):
        """动态调整alpha值
        
        随着训练进行，增加本地损失权重(个性化)
        """
        if round_idx < 20:
            # 前20轮更注重全局损失
            return max(0.2, base_alpha - 0.2)
        elif round_idx > 80:
            # 后期更注重本地损失
            return min(0.8, base_alpha + 0.2)
        else:
            # 中期使用基础值
            return base_alpha
    
    def _adjust_lambda(self, base_lambda, round_idx):
        """动态调整lambda值
        
        前期保持较高的特征对齐权重，后期逐渐降低
        """
        if round_idx < 30:
            # 前期强调特征对齐
            return min(0.5, base_lambda * 2.0)
        elif round_idx > 70:
            # 后期降低特征对齐约束
            return max(0.05, base_lambda * 0.5)
        else:
            # 中期使用基础值
            return base_lambda
    
    def to(self, device):
        """将损失函数移动到指定设备"""
        super().to(device)
        # 如果有projection_matrix，也将其移到对应设备
        for key, matrix in self.projection_matrix_cache.items():
            self.projection_matrix_cache[key] = matrix.to(device)
        return self